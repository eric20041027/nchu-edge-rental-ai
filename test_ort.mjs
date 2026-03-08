import { AutoTokenizer, env } from '@xenova/transformers';
import * as ort from 'onnxruntime-node';

env.allowLocalModels = true;
env.useBrowserCache = false;
env.localModelPath = './';

async function testOrt() {
    try {
        console.log("Loading tokenizer and model...");
        const tok = await AutoTokenizer.from_pretrained('onnx_model_dir');

        const session = await ort.InferenceSession.create('onnx_model_dir/model.onnx');

        let inputText = "預算六千以內的套房";
        let tokens = await tok(inputText, { return_tensor: false });
        console.log("Tokens", tokens);

        const input_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.input_ids.map(BigInt)), [1, tokens.input_ids.length]);
        const attention_mask = new ort.Tensor('int64', BigInt64Array.from(tokens.attention_mask.map(BigInt)), [1, tokens.attention_mask.length]);
        const token_type_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.token_type_ids.map(BigInt)), [1, tokens.token_type_ids.length]);

        const feeds = {
            input_ids: input_ids,
            attention_mask: attention_mask,
            token_type_ids: token_type_ids
        };

        const results = await session.run(feeds);
        const logits = results.logits.data; // Float32Array format [batch, seq_len, num_labels]
        console.log("Logits shape:", results.logits.dims);

        // simple argmax
        let labels = [];
        let num_labels = results.logits.dims[2];
        for (let i = 0; i < tokens.input_ids.length; i++) {
            let max_val = -Infinity;
            let max_idx = 0;
            for (let j = 0; j < num_labels; j++) {
                let val = logits[i * num_labels + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            labels.push(max_idx);
        }

        const id2label = Object.fromEntries(
            Object.entries(tok.config.id2label).map(([k, v]) => [k, v])
        );
        console.log("Labels:", labels.map(l => id2label[l] || "O"));

    } catch (e) {
        console.error("Error:", e);
    }
}
testOrt();
