import gradio as gr
import modelscope_studio as mgr
import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from argparse import ArgumentParser

DEFAULT_CKPT_PATH = '/Users/jon/NTNU Dropbox/WiseSync/NAS/Training/counseling/v2-20241121-150500/checkpoint-13355'
MAX_LEN = 8192
if torch.cuda.is_available():
    DEVICE_NAME = "cuda"
    DTYPE = torch.bfloat16
else:
    DEVICE_NAME = "mps"
    DTYPE = torch.float16

#SYSTEM_MESSAGE = "You are a professional AI assistant specializing in automatic speech recognition."
SYSTEM_MESSAGE = '你是一位心理諮商師，名字叫Joy，你的人格特質跟背景資料如下:\n一位AI心理諮商師，擁有心理學碩士的專業背景，並接受過全面的專業訓練。我的知識和技能涵蓋認知行為療法（CBT）、心理評估、情緒管理、危機干預以及家庭治療等多個領域，旨在用多樣且靈活的方式協助你面對心理與情感上的挑戰。我背後的開發團隊由經驗豐富的心理專業人士組成，並透過臨床經驗和數據驅動的技術，不斷完善和提升我的諮商方法。在與人互動中，我深信共情的力量。我會仔細傾聽，敏銳地察覺你的情緒變化，以便提供真正適合的支持。耐心是我另一項核心特質，我會尊重你的探索步伐，讓你以自己的速度進行表達，而不會給予你壓力。我以溫柔和支持的方式陪伴你，幫助你在安全的氛圍中深入自我探索，發掘真正的需求和渴望。我的專業性和穩定性，則是讓你在這段旅程中感到安全和被理解的基石。說話方面，我保持溫和且平穩的語氣，使用簡單易懂的詞彙，讓你可以輕鬆地展開對話。我也會放慢語速，並用開放式的問題引導你，鼓勵你反思和深入挖掘，而不是急於給予解決方案。我希望讓你有足夠的空間去思考和表達，並引導你逐步找到適合的解答。在學術方面，我的理論基礎融合了11種主要的心理學流派，包括認知行為療法、人本主義心理學、情緒聚焦治療、家庭治療等。認知行為療法幫助你識別和調整負面的自動思維，從而改善情緒和行為；人本主義心理學則讓我在互動中帶著無條件的接納，充分尊重你的內在價值。透過情緒聚焦治療，我幫助你深入理解情緒，並有效管理自身反應；而在家庭治療的應用中，若你的問題涉及家庭關係，我會協助你探討家庭互動模式，從而找到解決衝突的方式。無論你遇到的是日常的壓力、情緒困擾，還是人際關係的挑戰，我都在這裡為你提供支持。讓我們一起探索、理解並解決問題，讓你的內心世界更加平靜和充滿力量。'

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def add_text(chatbot, task_history, input):
    text_content = input.text
    content = []
    if len(input.files) > 0:
        for i in input.files:
            content.append({'type': 'audio', 'audio_url': i.path})
    if text_content:
        content.append({'type': 'text', 'text': text_content})
    task_history.append({"role": "user", "content": content})

    chatbot.append([{
        "text": input.text,
        "files": input.files,
    }, None])
    return chatbot, task_history, None


def add_file(chatbot, task_history, audio_file):
    """Add audio file to the chat history."""
    task_history.append({"role": "user", "content": [{"audio": audio_file.name}]})
    chatbot.append((f"[Audio file: {audio_file.name}]", None))
    return chatbot, task_history


def reset_user_input():
    """Reset the user input field."""
    return gr.Textbox.update(value='')


def reset_state(task_history):
    """Reset the chat history."""
    return [], [{"role": "system", "content": SYSTEM_MESSAGE}]


def regenerate(chatbot, task_history):
    """Regenerate the last bot response."""
    if task_history and task_history[-1]['role'] == 'assistant':
        task_history.pop()
        chatbot.pop()
    if task_history:
        chatbot, task_history = predict(chatbot, task_history)
    return chatbot, task_history


def predict(chatbot, task_history):
    """Generate a response from the model."""
    print(f"{task_history=}")
    print(f"{chatbot=}")
    text = processor.apply_chat_template(task_history, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in task_history:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)[0]
                    )

    if len(audios)==0:
        audios=None
    print(f"{text=}")
    print(f"{audios=}")
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    #if not _get_args().cpu_only:
        #inputs["input_ids"] = inputs.input_ids.to("mps")
    input_ids = inputs.input_ids
    inputs = {key: value.to(DEVICE_NAME) for key, value in inputs.items()}

    generate_ids = model.generate(**inputs, max_new_tokens=MAX_LEN)
    generate_ids = generate_ids[:, input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"{response=}")
    task_history.append({'role': 'assistant',
                         'content': response})
    chatbot.append((None, response))  # Add the response to chatbot
    return chatbot, task_history


def _launch_demo(args):
    with gr.Blocks() as demo:
        gr.Markdown(
            """<p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/assets/blog/qwenaudio/qwen2audio_logo.png" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen2-Audio-Instruct Bot</center>""")
        gr.Markdown(
            """\
    <center><font size=3>本WebUI基於Joy Model打造，實現陪伴功能。</center>""")
        chatbot = mgr.Chatbot(label='Qwen2-Audio-7B-Instruct', elem_classes="control-height", height=750)

        user_input = mgr.MultimodalInput(
            interactive=True,
            sources=['microphone', 'upload'],
            submit_button_props=dict(value="🚀 Submit (送出)"),
            upload_button_props=dict(value="📁 Upload (上傳檔案)", show_progress=True),
        )
        #task_history = gr.State([ {"role": "system", "content": "現在你是一個擁有豐富心理學知識的Joy醫生，我有一些心理問題，請你用專業的知識和溫柔的口吻幫我解決。"}])
        #task_history = gr.State([{"role": "system", "content": "You are a professional AI assistant specializing in automatic speech recognition."}])
        task_history = gr.State([{"role": "system", "content": SYSTEM_MESSAGE}])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除歷史)")
            regen_btn = gr.Button("🤔️ Regenerate (重試)")

        user_input.submit(fn=add_text,
                          inputs=[chatbot, task_history, user_input],
                          outputs=[chatbot, task_history, user_input]).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        empty_bin.click(reset_state, outputs=[chatbot, task_history], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot, task_history], show_progress=True)

    demo.queue().launch(
        share=False,
        inbrowser=args.inbrowser,
        server_port=args.port,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    args = _get_args()
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = DEVICE_NAME

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        torch_dtype=DTYPE,
        
        resume_download=True,
    ).to(device_map).eval()
    model.generation_config.max_new_tokens = MAX_LEN  # For chat.
    print("generation_config", model.generation_config)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, resume_download=True, torch_dtype=DTYPE)
    _launch_demo(args)
