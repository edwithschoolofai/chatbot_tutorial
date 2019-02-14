## 개요

이 코드는 Siraj Raval의 유튜브 영상 ["마케팅을 위한 챗봇"](https://youtu.be/PXJtFc8DjsE)에 대한 코드입니다.

# Deep Q&A
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

#### 목차

* [소개](#소개)
* [설치](#설치)
* [실행](#실행)
    * [챗봇](#챗봇)
    * [웹 인터페이스](#웹-인터페이스)
* [결과](#결과)
* [미리 학습된 모델](#미리-학습된-모델)
* [발전 가능성](#발전-가능성)

## 소개

이 프로젝트는 [A Neural Conversational Model](http://arxiv.org/abs/1506.05869)(혹은 구글 챗봇)의 결과를 재현합니다. 순환 신경망(seq2seq 모델)을 통해 다음 문장을 예측할 수 있으며, 파이썬과 텐서플로우를 사용합니다.

프로그램에서 말뭉치를 로딩하는 부분은 [macournoyer](https://github.com/macournoyer)의 Torch [neuralconvo](https://github.com/macournoyer/neuralconvo)에서 변형하였습니다.

현재 DeepQA는 다음과 같은 대화 말뭉치를 지원합니다.
 * [코넬 대학 영화 대사](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 말뭉치(기본). 이 저장소를 Clone하시면 자동으로 포함됩니다.
 * [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php)([Eschnou](https://github.com/eschnou)에게 감사드립니다). (노이즈가 더 많지만) 훨씬 큰 말뭉치입니다. 사용하려면 다음 [사용법](data/opensubs/)을 보시고 `--corpus opensubs` 플래그를 사용하세요.
 * 대법원 대화 자료 ([julien-c](https://github.com/julien-c)에게 감사드립니다). `--corpus scotus`를 쓰시면 사용 가능합니다. [사용법](data/scotus/)을 보시고 설치하세요.
 * [우분투 대화 말뭉치](https://arxiv.org/abs/1506.08909)([julien-c](https://github.com/julien-c)에게 감사드립니다). `--corpus ubuntu`를 쓰시면 사용 가능합니다. 다음 [사용법](data/ubuntu/)을 보고 설치하실 수 있습니다.
 * 여러분의 데이터([julien-c](https://github.com/julien-c)에게 감사드립니다)를 [다음](data/lightweight)과 같은 간단한 대화형식으로 쓰실 수 있습니다.

학습 속도를 올리기 위해 미리 학습된 단어 임베딩([Eschnou](https://github.com/eschnou)에게 감사드립니다)을 사용할 수도 있습니다. [더 보기](data/embeddings)

## 설치

이 프로그램은 다음과 같은 의존성이 필요합니다 (pip을 통해 쉽게 설치하실 수 있습니다: `pip3 install -r requirements.txt`):
 * python 3.5
 * tensorflow (v1.0로 실행되었습니다)
 * numpy
 * CUDA (GPU 사용 목적)
 * nltk (문장의 토큰화를 위한 자연어 처리 툴킷)
 * tqdm (진행 표시줄 목적)

nltk의 작동을 위해 추가 라이브러리를 설치해야할 수 있습니다.

```
python3 -m nltk.downloader punkt
```

코넬 데이터 세트는 이미 포함되어 있습니다. 다른 데이터 세트는 데이터 폴더(`data/`)내 readme를 참고하시기 바랍니다.

웹 인터페이스를 사용하시기 위해 다음 패키지가 필요합니다:
 * django (v1.10로 실행되었습니다)
 * channels
 * Redis ([여기](http://redis.io/topics/quickstart)를 참고하세요)
 * asgi_redis (v1.0 이상)

도커를 사용해서도 설치하실 수 있습니다. 자세한 내용은 [사용법](docker/README.md)을 참고하세요.

## 실행

### 챗봇

`main.py`를 실행하시면 모델을 학습시키실 수 있습니다. 학습 후 `main.py --test` (결과는 'save/model/samples_predictions.txt'에 저장됩니다) 나 `main.py --test interactive` (더 재미있습니다)를 통해 결과를 확인하실 수 있습니다.

도움이 될만한 플래그를 정리해 보았습니다. 더 많은 옵션을 알아보고 싶거나 도움이 필요하면 `python main.py -h`를 사용하세요:
 * `--modelTag <name>`: 현 모델의 학습/테스트 버전 구분을 위해 이름을 바꿀 수 있습니다.
 * `--keepAll`: 테스트 단계에서 스텝별 예측(학습이 진행되며 프로그램이 이름과 나이를 바꾸는 것이 재미있을 수 있습니다)을 보고 싶으실 경우 학습 단계에서 사용하세요. 경고 : `--saveEvery`를 높이지 않으면 저장 공간을 많이 차지할 수 있습니다.
 * `--filterVocab 20` 또는 `--vocabularySize 30000`: 어휘 크기를 제한하여 퍼포멘스와 메모리 사용을 최적화할 수 있습니다. 20회 미만 사용된 단어를 `<unknown>` 토큰으로 대체하고 최대 어휘 크기를 설정하세요.
 * `--verbose`: 테스트 단계에서 매 문장이 계산되는 즉시 출력됩니다.
 * `--playDataset`: 데이터 세트의 대화 샘플을 보여줍니다(실행할 유일한 명령일 경우 `--createDataset`와 함께 사용할 수 있습니다).

[텐서보드](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)를 통해 계산 그래프와 비용을 보시려면 `tensorboard --logdir save/`를 실행하세요.

기본적으로 네트워크 구조는 두 개의 LSTM(은닉층의 크기=256)과 임베딩(어휘 개수=32)을 가진 표준 인코더/디코더입니다. ADAM을 통해 네트워크를 학습시킵니다. 문장 당 최대 단어 개수는 10이지만, 그 이상 또한 가능합니다.

### 웹 인터페이스

학습 한 후 더 친숙한 인터페이스를 통해 프로그램과 채팅이 가능합니다. 서버는 `save/model-server/model.ckpt`에 저장된 모델을 사용할 것입니다. 처음 사용하실 때 다음과 같이 설정합니다:

```bash
export CHATBOT_SECRET_KEY="my-secret-key"
cd chatbot_website/
python manage.py makemigrations
python manage.py migrate
```

로컬 서버에서 실행할 경우 다음 명령어를 입력하세요:

```bash
cd chatbot_website/
redis-server &  # Launch Redis in background
python manage.py runserver
```

서버 실행 후, [http://localhost:8000/](http://localhost:8000/)에서 인터페이스를 보실 수 있습니다. 프로그램을 서버에 배포하려면 `python manage.py runserver 0.0.0.0`을 실행하세요. [더보기](https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/)

[![Chatbot web interface](chatbot_miniature.png)](http://e-pot.xyz/cv/chatbot.png)

## 결과

놀랍게도, 한두 시간만 학습(GeForce GT 740 사용)을 해도 어느 정도의 결과를 얻을 수 있습니다. 단시간에 이러한 결과를 얻으려면 입력 시퀀스의 단어를 5개로, 출력을 3개(`go`와 `eos`토큰 또한 추가로 포함)로 제한하고 작은 임베딩 사이즈(대략 25)를 사용해야 합니다. 이후 출력값과 입력값의 길이가 같아지도록 코드를 수정했지만 [이 버전](https://github.com/Conchylicultor/DeepQA/tree/92863e2929580818f866f16969909fe2093d41d1)을 통해 처음 결과를 재현하실 수 있습니다.
물론, 이러한 초기 단계의 네트워크는 아직 많은 대화를 할 수 없습니다:

    Q: Hi (안녕)
    A: Hi. (안녕.)

    Q: What is your name ? (이름이 뭐니?)
    A: Laura. (Laura.)

    Q: What does that mean ? (그게 무슨 뜻이야?)
    A: I dunno. (나도 몰라.)

    Q: How old are you ? (몇 살이니?)
    A: thirty-five. (서른 다섯.)

    Q: Will Google hire me ? (구글이 나를 채용할까?)
    A: No. (아니.)

    Q: Tell me the alphabet (알파벳을 말해줘)
    A: Fuck you. (비속어)

    Q: That's not nice (그거 안 좋은 행동이야.)
    A: Yeah. (응.)

다음과 같은 실패 케이스 또한 있습니다:

    Q: Who is Laura ? (Laura가 누구니?)
    A: My brother. (내 형제.)

    Q: Say 'goodbye' ('안녕'이라고 말해봐)
    A: Alright. (그래.)

    Q: What is cooking ? (요리가 뭐야?)
    A: A channel. (채널.)

    Q: Can you say no ? (아니라고 말할 수 있니?)
    A: No. (아니.)

    Q: Two plus two (2 더하기 2)
    A: Manny... (마않은...)

문장이 길어질수록 네트워크는 훨씬 느리게 학습됩니다. 네트워크를 maxLength=10으로 250,000회 학습한 후에도 대부분 회피형 응답(예: 개방형 질문에 '잘 모르겠습니다'로 대답)을 합니다. 하지만 근본적인 의미를 어느정도 파악한 것으로 보입니다.

    Q: When are the fireworks ? (불꽃놀이 언제해?)
    A: Tomorrow. (내일.)

    Q: What color ? (무슨 색?)
    A: Gray. (회색.)

    Q: Which apple ? (어떤 사과?)
    A: That's the first one. (저게 첫 번째야.)

    Q: What time is it ? (지금 몇시야?)
    A: Eleven o'clock. (11시.)

    Q: Who is Skywalker ? (스카이워커가 누구야?)
    A: She's a freelance bio-exorcist. (그녀는 프리랜서 생물학적 퇴마사야.)

    Q: Who is John ? (존이 누구야?)
    A: The old man. (그 노인.)

    Q: Who is Jenny ? (제니가 누구야?)
    A: The man who killed herself. (그녀 스스로 자살한 남자야.)

    Q: When were you born ? (언제 태어났니?)
    A: Soon. (곧.)

마지막 질문에서 볼 수 있듯이, 네트워크는 성별과 현재/미래를 아직 구분하기 어려워합니다. 더 심도있는 철학적인 질문 또한 어느 정도 비슷한 결과를 보입니다.

    Q: What is the purpose of being intelligent ? (지능의 목적은 뭘까?)
    A: The man are obsolete. (인간은 구식이야.)

    Q: What is immoral ? (부도덕이란 뭘까?)
    A: The truth. (진리.)

    Q: What is morality ? (도덕은 뭘까?)
    A: The worst. (최악.)

    Q: What is the purpose of existence ? (존재의 이유는 뭘까?)
    A: It 's private. (사적이야.)

    Q: What happens if machines can think ? (기계가 생각할 수 있다면 어떻게 될까?)
    A: I think I'll just lie with you. (그냥 너한테 거짓말하기로 했어.)

이 모델은 임베딩 사이즈와 학습용 말뭉치 크기에 따른 한계가 있습니다. 모델의 '생각할 수 있는 벡터'가 너무 작아서 다음과 같은 질문에 제한된 답을 제공하는 것으로 보입니다.

    Q: My name is David. What is my name ? (내 이름은 데이비드야. 내 이름이 뭐니?)
    A: Debbie. (데비.)

    Q: My name is John. What is my name ? (내 이름은 존이야. 내 이름이 뭐니?)
    A: Carl Smith. (칼 스미스.)

    Q: How many legs does a cat have ? (고양이는 다리가 몇 개 있어?)
    A: Four. (네 개.)

    Q: How many legs does a spider have ? (거미는 다리가 몇 개 있어?)
    A: Four. (네 개.)

    Q: Do you prefer football or soccer ? (풋볼과 축구 중에 뭐가 더 좋아?)
    A: Whiskey. (위스키.)

또한 때로 질문과 무관한 답을 하는것으로 보아, 학습 과정에서 오버피팅이 되는 것으로 보입니다. 드롭아웃을 조금 추가하면 문제가 어느 정도 해결될 수 있습니다.

    Q: I play tennis. What do I play ? (나 테니스 할꺼야. 내가 뭐 한다고?)
    A: I'm going to get laid. (나 잠자리 가질거야.)

    Q: What color is the sky ? (하늘은 무슨색이니?)
    A: One of the deputies is everywhere. (대리인 중 한명이 어디든지 있어.)

## 미리 학습된 모델

기본 말뭉치에 의해 미리 학습된 [모델](https://drive.google.com/file/d/0Bw-phsNSkq23OXRFTkNqN0JGUU0/view?usp=sharing)을 사용하실 수 있습니다. 다음과 같이 사용하실 수 있습니다:
 1. `DeepQA/save/` 내부 zip 파일의 압축을 풉니다.
 2. `save/model-pretrainedv2/dataset-cornell-old-lenght10-filter0-vocabSize0.pkl`에서 전처리 된 데이터셋을 `data/samples/`로 복사합니다.
 3. `./main.py --modelTag pretrainedv2 --test interactive`를 실행합니다.

Nicholas C.의 도움으로, [여기서](https://drive.google.com/drive/folders/0Bw-phsNSkq23c29ZQ2N6X3lyc1U?usp=sharing) ([원본](https://mcastedu-my.sharepoint.com/personal/nicholas_cutajar_a100636_mcast_edu_mt/_layouts/15/guestaccess.aspx?folderid=077576c4cf9854642a968f67909380f45&authkey=AVt2JWMPkf2R_mWBpI1eAUY)) 다양한 데이터셋을 위한 미리 학습 된 모델(텐서플로 v1.2와 호환 가능)을 사용하실 수 있습니다. 이 폴더에는 전처리 된 Cornell, OpenSubtitles, Ubuntu and Scotus 데이터셋이 포함되어 있습니다 (`data/samples/`으로 옮기기 위해서요). 직접 데이터셋을 처리하고 싶지 않으실 경우 사용하실 수 있습니다.

성능이 좋은 GPU를 사용하신다면 변수나 말뭉치를 간단히 조정하여 더 좋은 모델로 학습시킬 수 있습니다. 제 경험으로는 학습률이나 드롭아웃 비율이 결과에 가장 큰 영향을 미치는 것으로 보였습니다. 혹시 모델을 공유하고 싶으시다면 부담 없이 저에게 연락하세요. 여기에 모델을 공유하겠습니다.

## 발전 가능성


모델의 크기나 깊이 외에도 다양한 방법으로 시도해 보실 수 있습니다. 구현하시면 부담 없이 pull request를 보내주세요. 다음과 같은 예시 아이디어들이 있습니다:

*  현 모델의 예측은 결정론적이므로(가장 가능성이 높은 대답을 출력) 똑같은 질문에 항상 동일한 대답을 할 것입니다. 샘플링 메커니즘을 추가한다면 훨신 다양하고 (더 재미있을 수 있는) 답을 제공할 수 있습니다. 가장 쉬운방법은 SoftMax 확률 분포에서 예측 된 다음 단어를 샘플링 하는 것입니다. `tf.nn.seq2seq.rnn_decoder`의`loop_function`을 사용하면 많이 어렵지 않을 것입니다. 이 후 SoftMax를 가지고 실험하시며 더 보수적이거나 신기한 예측을 생성시킬 수 있습니다.

* 어텐션을 추가하면 더 긴 문장에 특히 더 예측을 향상시킬 수 있습니다. 'embedding_rnn_seq2seq`을 `model.py`의 `embedding_attention_seq2seq`로 바꾸면 간단합니다.


* 데이터가 많으면 보통 더 결과가 좋습니다. 더 큰 말뭉치에 학습을 시키는 것이기 때문입니다. [Reddit 댓글 데이터셋](https://www.reddit.com/r/datasets/comments/59039y/updated_reddit_comment_dataset_up_to_201608/)이 현재 가장 큰 말뭉치로 보입니다 (이 프로그램에 사용하기엔 크기가 커서 부적절합니다). 혹은 말뭉치를 만들 때 각 학습 샘플의 문장을 나눠서 데이터셋의 크기를 인위적으로 늘릴수있습니다 (예: 샘플 `Q:문장 1. 문장 2 => A: 문장 X. 문장 Y`로 샘플 세 개를 만들 수 있습니다: `Q: 문장1. 문장 2 => A:문장 X`, `Q:문장 2 => A:문장 X. 문장 Y`, `Q:문장 2 => A:문장 X`. 경고 : `Q:문장 1. => A:문장 X.`와 같은 조합은 `2 => X` 처럼 질문에서 답변으로의 전환을 깨뜨리기 때문에 작동되지 않습니다)
* 테스트 곡선은 제 [음악 생성](https://github.com/Conchylicultor/MusicGenerator) 프로젝트에서처럼 모니터 되는 것이 좋습니다. 이럴 경우 드롭아웃이 오버피팅에 미치는 효과를 관찰하실 수 있습니다. 일단은 단순히 각 학습 스텝에서 예측된 결과를 체크하며 진행합니다. 
* 현재 모든 질문은 서로 독립적입니다. 질문을 서로 연결하려면 대답을 하기 전에 이전 질문을 모두 입력하고 인코더가 응답하게 하는 것이 가장 간단합니다. 마지막 인코더에서 캐시를 저장하면 매번 다시 계산하지 않아도 됩니다. 정확성을 높이려면 개별 질문/대답 보다는 전체 대화를 학습시키는 것이 좋습니다. 또한 이전 대화를 인코더에 입력할 때 `<Q>`, `<A>` 토큰을 추가한다면 인코더가 화자가 바뀔 때를 파악할 수 있습니다. 간단한 seq2seq 모델이 문장 사이의 장기 의존성을 포착하기 충분할지는 모르겠습니다. 비슷한 길이의 입력 문장을 그룹화하는 버킷 시스템을 추가하면 학습 속도가 크게 향상 될 수 있습니다.
   
## 감사의 말 

[Conchylucultor](https://github.com/Conchylicultor/DeepQA)에 감사의 말씀을 전합니다. 이 코드는 제가 조금만 변형한 것입니다.
