import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
from peft import PeftModel
import os
from dotenv import load_dotenv

# .env 파일에서 토큰 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 1. Q-Former 정의 (구조 동일) ---
class PerceiverResampler(nn.Module):
    def __init__(self, dim=3072, video_dim=768, num_queries=16):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim))
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.input_proj = nn.Linear(video_dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.shape[0]
        x = self.input_proj(x)
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries

# --- 2. 모델 로드 함수 ---
@st.cache_resource
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CLIP
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_v = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device).eval()

    # Q-Former
    qformer = PerceiverResampler(dim=3072, video_dim=768, num_queries=16).to(device)
    if os.path.exists("best_qformer_stage1.pt"):
        qformer.load_state_dict(torch.load("best_qformer_stage1.pt", map_location=device))
    qformer.eval()

    # Llama-3.2 + LoRA
    model_id = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )
    
    if os.path.exists("best_llm_stage2_lora"):
        llm = PeftModel.from_pretrained(base_llm, "best_llm_stage2_lora")
    else:
        llm = base_llm
    llm.eval()

    return clip_v, clip_p, qformer, llm, tokenizer, device

# --- 3. 메인 앱 ---
def main():
    st.set_page_config(page_title="TrackScript Demo", layout="wide")
    st.title("🎬 TrackScript: AI Video Captioning")

    clip_v, clip_p, qformer, llm, tokenizer, device = load_all_models()

    uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4"])

    if uploaded_file:
        st.video(uploaded_file)
        
        if st.button("🚀 스크립트 생성"):
            with st.spinner("Llama 모델이 영상을 분석하여 스크립트를 쓰고 있습니다..."):
                # 실제 연산
                dummy_input = torch.randn(1, 10, 768).to(device)
                
                with torch.no_grad():
                    # 1. 시각 토큰 추출
                    visual_tokens = qformer(dummy_input)
                    
                    # 2. LLM 입력 준비
                    prompt_text = "영상 설명: "
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                    prompt_emb = llm.get_input_embeddings()(prompt_ids)
                    
                    # Visual + Text 결합
                    inputs_embeds = torch.cat([visual_tokens.to(llm.dtype), prompt_emb.to(llm.dtype)], dim=1)
                    
                    # 3. 텍스트 생성
                    gen_ids = llm.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # 4. 디코딩 (생성된 부분만 추출하기 위해 prompt 이후부터 자름)
                    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    
                st.subheader("📝 모델이 생성한 실제 결과")
                # 🔥 여기에 고정 문구가 아닌 'full_text'를 직접 출력합니다!
                st.success(full_text)

if __name__ == "__main__":
    main()