import { AutoTokenizer, env } from '@xenova/transformers';

env.allowLocalModels = true;
env.useBrowserCache = false;
env.localModelPath = './';

async function testTok() {
    try {
        console.log("Loading tokenizer...");
        const tok = await AutoTokenizer.from_pretrained('onnx_model_dir');
        const out = await tok("預算六千", { return_tensor: false });
        console.log("Success!", out);
    } catch (e) {
        console.error("Tokenizer Error:", e);
    }
}
testTok();
