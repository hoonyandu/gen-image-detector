# gen-image-detector
> Verify generated ai image with model with Gradio UI.
> 
> 기간: 2025.10.15 ~ 2025.10.24</br>
> 참여자: [hoonyandu](https://github.com/hoonyandu), [sominify](https://github.com/sominify)</br>
> 참고자료: [HuggingFace] [Ateeqq/ai-vs-human-image-detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector)</br>

## Structure
```text
.
└── gen-image-detection
    ├── README.md
    ├── image           # Examples 로 사용할 이미지
    │
    ├── scripts         # local docker 환경 구성을 위한 스크립트
    │   └── docker      # docker # 컨테이너, 이미지 정리 스크립트
    │
    ├── src
    │   ├── domain      # 모델, 시각화 클래스
    │   ├── services    # 모델 예측, 시각화 실행
    │   │
    │   ├── __init__.py
    │   ├── app.py      # Gradio 실행
    │   └── settings.py
    │
    ├── test            # local 환경 기능 테스트 실행
    ├── deploy.sh       # local(cpu) 도커 환경 테스트 실행 스크립트
    ├── docker-compose.yml
    ├── Dockerfile
    └── requirements.txt
```
</br>

## Environment

### Local
<pre>
<span style="color:blue">Python: </span><span style="color:green">3.12.7</span>
<span style="color:blue">Chip: </span><span style="color:green">Apple M3 Pro</span>
<span style="color:blue">OS: </span><span style="color:green">macOS Sequoia 15.5</span>
</pre>

## Usage
```shell
# local Gradio app 실행
$ python src/app.py

# cpu 환경 도커 Gradio app 실행
$ ./deploy.sh
```