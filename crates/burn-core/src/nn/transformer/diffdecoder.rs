use alloc::vec::Vec;

use super::{PositionWiseFeedForward, PositionWiseFeedForwardConfig};

use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::tensor::Bool;
use crate::{
    self as burn,
    nn::{attention::MhaCache, cache::TensorCache, Initializer},
};
use crate::{
    config::Config,
    nn::{
        attention::{MhaInput, DiffMhaInput, MultiHeadAttention, MultiHeadAttentionConfig, DiffMultiHeadAttention, DiffMultiHeadAttentionConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, transformer::decoder::*,
    },
    tensor::{backend::Backend, Tensor},
};




#[derive(Module, Debug)]
pub struct DiffTransformerDecoderLayer<B: Backend> {
    cross_attn: DiffMultiHeadAttention<B>,
    self_attn: DiffMultiHeadAttention<B>,
    pwff: PositionWiseFeedForward<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    norm_3: LayerNorm<B>,
    dropout: Dropout,
    norm_first: bool,
}

impl<B: Backend> DiffTransformerDecoderLayer<B> {
    fn new(config: &TransformerDecoderConfig, device: &B::Device) -> Self {
        let self_attn = DiffMultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_quiet_softmax(config.quiet_softmax)
            .init(device);

        let cross_attn = DiffMultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_quiet_softmax(config.quiet_softmax)
            .init(device);
        let norm_1 = LayerNormConfig::new(config.d_model).init(device);
        let norm_2 = LayerNormConfig::new(config.d_model).init(device);
        let norm_3 = LayerNormConfig::new(config.d_model).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_dropout(config.dropout)
            .init(device);

        Self {
            cross_attn,
            self_attn,
            norm_1,
            norm_2,
            norm_3,
            pwff,
            dropout,
            norm_first: config.norm_first,
        }
    }

    /// Applies the TransformerDecoder forward pass to the input tensor.
    fn forward(&self, mut input: TransformerDecoderInput<B>) -> TransformerDecoderInput<B> {
        // Self attention residual path.
        let x = input.target;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = self.norm_3.forward(residual_path);
        }

        // Self attention.
        let mut self_attn_input = DiffMhaInput::self_attn(residual_path, 0.5);
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = self.self_attn.forward(self_attn_input).context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Cross attention residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            self.norm_1.forward(x.clone())
        } else {
            x = self.norm_1.forward(x);
            x.clone()
        };

        // Cross attention.
        let mut cross_attn_input =
            DiffMhaInput::new(residual_path, input.memory.clone(), input.memory.clone(), 0.5);
        if let Some(mask_pad) = &input.memory_mask_pad {
            cross_attn_input = cross_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            cross_attn_input = cross_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = self.cross_attn.forward(cross_attn_input).context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            self.norm_2.forward(x.clone())
        } else {
            x = self.norm_2.forward(x);
            x.clone()
        };

        let residual_path = self.pwff.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = self.norm_3.forward(x)
        }

        input.target = x;
        input
    }

    fn forward_autoregressive_inference(
        &self,
        mut input: TransformerDecoderInput<B>,
        cache: &mut TransformerDecoderLayerAutoregressiveCache<B>,
    ) -> TransformerDecoderInput<B> {
        // Self attention residual path.
        let x = input.target;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = cache
                .norm_3
                .forward_autoregressive(residual_path, 1, |x| self.norm_3.forward(x));
        }

        // Self attention.
        let mut self_attn_input = DiffMhaInput::self_attn(residual_path, 0.5);
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = self
            .self_attn
            .forward_cache(self_attn_input, &mut cache.self_attn)
            .context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Cross attention residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            cache
                .norm_1
                .forward_autoregressive(x.clone(), 1, |x| self.norm_1.forward(x))
        } else {
            x = cache
                .norm_1
                .forward_autoregressive(x, 1, |x| self.norm_1.forward(x));
            x.clone()
        };

        // Cross attention.
        let mut cross_attn_input =
            DiffMhaInput::new(residual_path, input.memory.clone(), input.memory.clone(), 0.5);
        if let Some(mask_pad) = &input.memory_mask_pad {
            cross_attn_input = cross_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            cross_attn_input = cross_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = self
            .cross_attn
            .forward_cache(cross_attn_input, &mut cache.cross_attn)
            .context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            cache
                .norm_2
                .forward_autoregressive(x.clone(), 1, |x| self.norm_2.forward(x))
        } else {
            x = cache
                .norm_2
                .forward_autoregressive(x, 1, |x| self.norm_2.forward(x));
            x.clone()
        };

        let residual_path = cache
            .pwff
            .forward_autoregressive(residual_path, 1, |x| self.pwff.forward(x));
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = cache
                .norm_3
                .forward_autoregressive(x, 1, |x| self.norm_3.forward(x))
        }

        input.target = x;
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Distribution;
    use crate::{nn::attention::generate_autoregressive_mask, TestBackend};

    #[test]
    fn test_autoregressive_norm_last() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        TestBackend::seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers)
                .with_norm_first(false),
        )
    }

    #[test]
    fn test_autoregressive_norm_first() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        TestBackend::seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers).with_norm_first(true),
        )
    }

    fn test_autoregressive(config: TransformerDecoderConfig) {
        let device = Default::default();
        let [batch_size, seq_length, d_model] = [3, 4, config.d_model];
        let transformer = config.init(&device);

        let memory = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let target = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &target.device());
        let input = TransformerDecoderInput::new(target.clone(), memory.clone())
            .target_mask_attn(mask_attn);

        // Normal forward using masking.
        let output_1 = transformer.forward(input);

        // Forward using the autoregressive cache.
        let mut output_2 = Vec::new();
        let mut cache = transformer.new_autoregressive_cache();

        for i in 1..seq_length + 1 {
            let target = target.clone().slice([0..batch_size, 0..i, 0..d_model]);

            let mask_attn = generate_autoregressive_mask(batch_size, i, &target.device());
            let input = TransformerDecoderInput::new(target.clone(), memory.clone())
                .target_mask_attn(mask_attn);
            let next_tok = transformer // Greedy sampling
                .forward_autoregressive_inference(input, &mut cache)
                .slice([0..batch_size, i - 1..i, 0..d_model]);
            output_2.push(next_tok);
        }

        let output_2 = Tensor::cat(output_2, 1);

        // Should produce the same tokens.
        output_1
            .into_data()
            .assert_approx_eq(&output_2.into_data(), 3);
    }

    #[test]
    fn display() {
        let config = TransformerDecoderConfig::new(2, 4, 2, 3);
        let transformer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", transformer),
            "TransformerDecoder {d_model: 2, d_ff: 4, n_heads: 2, n_layers: 3, \
            dropout: 0.1, norm_first: false, quiet_softmax: false, params: 246}"
        );
    }
}