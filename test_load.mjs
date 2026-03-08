import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = true;
env.useBrowserCache = false;
env.localModelPath = './';

async function test() {
    try {
        console.log("Loading model...");
        const pipe = await pipeline('token-classification', 'onnx_model_dir', {
            quantized: false
        });
        const out = await pipe("預算六千以內套房");
        console.log("Success!", out);
    } catch (e) {
        console.error("Error details:", e);
    }
}
test();
