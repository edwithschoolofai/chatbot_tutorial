## 간단한 설명

이 코드는 Siraj Raval 의 유투브 비디오 ["마케팅을 위한 챗봇"](https://youtu.be/PXJtFc8DjsE)에 설명 된 코드입니다.

# Deep Q&A
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

#### 목차

* [Presentation](#presentation)
* [Installation](#installation)
* [Running](#running)
    * [Chatbot](#chatbot)
    * [Web interface](#web-interface)
* [Results](#results)
* [Pretrained model](#pretrained-model)
* [Improvements](#improvements)
* [Upgrade](#upgrade)

## Presentation

이 프로젝트는 [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (혹은 구글 챗봇)의 결과를 재현합니다. 코드를 통해 순환신경망(RNN의 seq2seq 모델)을 통해 다음 문장을 예측하실 수 있으며, 파이썬과 텐서플로우를 사용합니다.

프로그램에서 말뭉치를 로딩하는 부분은 [macournoyer](https://github.com/macournoyer)의 Torch [neuralconvo](https://github.com/macournoyer/neuralconvo)에서 변형하였습니다.

현재 DeepQA는 다음과 같은 대화 말뭉치를 지원합니다.
 * [코넬 영화 대사](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 말뭉치 (기본). 이 저장소를 복제하시면 자동으로 포함됩니다.
 * [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php) ([Eschnou](https://github.com/eschnou)에게 감사드립니다). (잡음이 더 많지만) 훨씬 큰 말뭉치입니다. 사용하려면 [다음 사용법](data/opensubs/) 을 보시고 `--corpus opensubs`를 쓰세요.
 * 대법원 대화 자료 ([julien-c](https://github.com/julien-c)에게 감사드립니다). `--corpus scotus`를 쓰시면 사용 가능합니다. See the [instructions](data/scotus/) for installation.
 * [우분투 대화 말뭉치](https://arxiv.org/abs/1506.08909) ([julien-c](https://github.com/julien-c)에게 감사드립니다). `--corpus ubuntu`를 쓰시면 사용 가능합니다. 다음 [사용법](data/ubuntu/)을 보고 설치하실 수 있습니다.
 * 여러분의 데이터([julien-c](https://github.com/julien-c)에게 감사드립니다)를 다음과 같은 [간단한 대화형식](data/lightweight)으로 쓰실 수 있습니다.

학습의 속도를 올리기 위해 미리 학습된 단어 임베딩([Eschnou](https://github.com/eschnou)에게 감사드립니다)를  수 있습니다. [](data/embeddings).

## 설치

이 프로그램은 다음과 같은 의존성을 요구합니다 (pip을 통해 쉽게 설치하실 수 있습니다: `pip3 install -r requirements.txt`):
 * 파이썬 3.5
 * 텐서플로우 (v1.0로 실행되었습니다)
 * 넘파이
 * CUDA (GPU 사용 목적)
 * nltk (문장의 토큰화를 위한 자연어 처리 툴킷)
 * tqdm (진행 표시줄 목적)

nltk의 작동을 위해 추가 의존성을 설치해야 하실 수 있습니다.

```
python3 -m nltk.downloader punkt
```

코넬 데이터 세트는 이미 포함되어 있습니다. 다른 데이터 세트는 데이터 폴더(`data/`)내 readme를 참고하시기 바랍니다.

웹 인터페이스를 사용하시기 위해 다음 패키지가 필요합니다:
 * 장고 (v1.10로 실행되었습니다)
 * channels
 * Redis ([여기](http://redis.io/topics/quickstart)를 참고하세요)
 * asgi_redis (v1.0 이상)

도커를 사용해서도 설치를 하실 수 있습니다. 자세한 [사용법](docker/README.md)을 참고하세요.

## 실행

### 챗봇

`main.py`를 실행하시면 모델을 학습시키실 수 있습니다. 학습 후 `main.py --test` (결과는 'save/model/samples_predictions.txt'에 저장됩니다) 나 `main.py --test interactive` (더 재미있습니다) 를 통해 결과를 확인하실 수 있습니다.

Here are some flags which could be useful. For more help and options, use `python main.py -h`:
 * `--modelTag <name>`: 현 모델의 학습/테스트 버젼 구분을 위해 이름을 바꿀 수 있습니다.
 * `--keepAll`: 테스트 단계에서 스텝별 예측(학습이 진행되며 프로그램이 이름과 나이를 바꾸는 것이 재미있을 수 있습니다)을 보시고 싶으실 경우 학습 단계에서 사용하세요. 경고 : `--saveEvery` 를 높이지 않으면 저장 공간을 많이 차지할 수 있습니다.
 * `--filterVocab 20` 또는 `--vocabularySize 30000`: 어휘 크기를 제한하여 퍼포멘스와 메모리 사용을 최적화 하실 수 있습니다. 20회 미만 사용된 단어를 `<unknown>` 토큰으로 대체하시고 최대 어휘 크기를 설정하.
 * `--verbose`: 테스트 단계에서 때 매 문장이 계산되는 즉시 출력됩니다.
 * `--playDataset`: 데이터 세트의 대화 샘플을 보여줍니다 (수행 하실 유일한 명령일 경우 `--createDataset` 와 함께 사용하실 수 있습니다).

[텐서보드](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)를 통해 계산 그래프와 비용을 보시려면 `tensorboard --logdir save/`를 실행하세요.

기본적으로 네트워크 구조는 두 개의 LSTM(은닉층의 크기=256)과 임베딩(어휘 개수=32)을 가진 표준 인코더/디코더입니다. ADAM을 통해 네트워크를 학습시킵니다. 문장 당 최대 단어 개수는 10이지만, 그 이상 또한 .

### 웹 인터페이스

학습 한 후 더 친숙한 인터페이스를 통해 프로그램과 채팅이 가능합니다. 서버는 `save/model-server/model.ckpt`에 저장된 모델을 사용할 것입니다. 처음 사용하실 때 다음과 같이 환경 설정을 하시면 됩니다:

```bash
export CHATBOT_SECRET_KEY="my-secret-key"
cd chatbot_website/
python manage.py makemigrations
python manage.py migrate
```

로컬 서버에서 론치하실 경우 다음 명령어를 입력하세요:

```bash
cd chatbot_website/
redis-server &  # Launch Redis in background
python manage.py runserver
```

서버 론치 후, [http://localhost:8000/](http://localhost:8000/)에서 인터페이스가 보실 수 있습니다. 프로그램을 서버에서 디플로이 하려면 `python manage.py runserver 0.0.0.0` 을 실행하세요. [더보기](https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/).

[![Chatbot web interface](chatbot_miniature.png)](http://e-pot.xyz/cv/chatbot.png)

## 결과

신기하게도, 한 두 시간만 학습 (GeForce GT 740로 학습)을 해도 어느정도의 결과를 얻을 수 있습니다. 단 시간에 이러한 결과를 얻으려면 입력 시퀀스의 단어를 5개로, 출력을 3개(`go` 와 `eos`토큰 또한 추가로 포함)로 제한하고 작은 임베딩 사이즈(대략 25)를 사용해야 합니다. 이후 출력값과 입력값이 어느정도 비슷한 길이가 되도록 코드를 수정했지만 [이 버젼](https://github.com/Conchylicultor/DeepQA/tree/92863e2929580818f866f16969909fe2093d41d1)을 통해 처음 결과를 재현하실 수 있습니다.
물론, 이러한 초기 단계의 네트워크는 아직 많은 대화를 할 수 없습니다:

    Q: Hi
    A: Hi.

    Q: What is your name ?
    A: Laura.

    Q: What does that mean ?
    A: I dunno.

    Q: How old are you ?
    A: thirty-five.

    Q: Will Google hire me ?
    A: No.

    Q: Tell me the alphabet
    A: Fuck you.

    Q: That's not nice
    A: Yeah.

다음과 같은 실패 케이스 또한 있습니다:

    Q: Who is Laura ?
    A: My brother.

    Q: Say 'goodbye'
    A: Alright.

    Q: What is cooking ?
    A: A channel.

    Q: Can you say no ?
    A: No.

    Q: Two plus two
    A: Manny...

문장이 길어질수록 네트워크는 훨씬 느리게 학습됩니다. 네트워크를 maxLength = 10으로 250000회 학습한 후에도 대부분 회피형 응답(예: 개방형 질문에 '잘 모르겠습니다'로 대답)을 합니다. 하지만 근본적인 의미를 어느정도 파악한 것으로 보입니다.

    Q: When are the fireworks ?
    A: Tomorrow.

    Q: What color ?
    A: Gray.

    Q: Which apple ?
    A: That's the first one.

    Q: What time is it ?
    A: Eleven o'clock.

    Q: Who is Skywalker ?
    A: She's a freelance bio-exorcist.

    Q: Who is John ?
    A: The old man.

    Q: Who is Jenny ?
    A: The man who killed herself.

    Q: When were you born ?
    A: Soon.

마지막 질문에서 볼 수 있듯이, 네트워크는 성별과 현재/미래를 아직 구분하기 어려워합니다. 더 심도있는 철학적인 질문 또한 어느정도 비슷한 결과를 보입니다.

    Q: What is the purpose of being intelligent ?
    A: The man are obsolete.

    Q: What is immoral ?
    A: The truth.

    Q: What is morality ?
    A: The worst.

    Q: What is the purpose of existence ?
    A: It 's private.

    Q: What happens if machines can think ?
    A: I think I'll just lie with you.

이 모델은 임베딩 사이즈와 학습용 말뭉치 크기에 의해 상당히 제한됩니다. 모델의 '생각할 수 있는 벡터'가 너무 작아서 다음과 같은 질문에 제한된 답을 제공하는 것으로 보입니다.

    Q: My name is David. What is my name ?
    A: Debbie.

    Q: My name is John. What is my name ?
    A: Carl Smith.

    Q: How many legs does a cat have ?
    A: Four.

    Q: How many legs does a spider have ?
    A: Four.

    Q: Do you prefer football or soccer ?
    A: Whiskey.

또한 때로 질문과 무관한 답을 하는것으로 보아, 학습 과정에서 오버피팅이 되는 것으로 보입니다. 드롭 아웃을 조금 추가하면 문제가 어느정도 해결할 수 있습니다.

    Q: I play tennis. What do I play ?
    A: I'm going to get laid.

    Q: What color is the sky ?
    A: One of the deputies is everywhere.

## Pretrained model

You can find a pre-trained model [here](https://drive.google.com/file/d/0Bw-phsNSkq23OXRFTkNqN0JGUU0/view?usp=sharing), trained of the default corpus. To use it:
 1. Extract the zip file inside `DeepQA/save/`
 2. Copy the preprocessed dataset from `save/model-pretrainedv2/dataset-cornell-old-lenght10-filter0-vocabSize0.pkl` to `data/samples/`.
 3. Run `./main.py --modelTag pretrainedv2 --test interactive`.

Thanks to Nicholas C., [here](https://drive.google.com/drive/folders/0Bw-phsNSkq23c29ZQ2N6X3lyc1U?usp=sharing) ([original](https://mcastedu-my.sharepoint.com/personal/nicholas_cutajar_a100636_mcast_edu_mt/_layouts/15/guestaccess.aspx?folderid=077576c4cf9854642a968f67909380f45&authkey=AVt2JWMPkf2R_mWBpI1eAUY)) are some additional pre-trained models (compatible with TF 1.2) for diverse datasets. The folder also contains the pre-processed dataset for Cornell, OpenSubtitles, Ubuntu and Scotus (to move inside `data/samples/`). Those are required is you don't want to process the datasets yourself.

If you have a high-end GPU, don't hesitate to play with the hyper-parameters/corpus to train a better model. From my experiments, it seems that the learning rate and dropout rate have the most impact on the results. Also if you want to share your models, don't hesitate to contact me and I'll add it here.

## Improvements

In addition to trying larger/deeper model, there are a lot of small improvements which could be tested. Don't hesitate to send a pull request if you implement one of those. Here are some ideas:

* For now, the predictions are deterministic (the network just take the most likely output) so when answering a question, the network will always gives the same answer. By adding a sampling mechanism, the network could give more diverse (and maybe more interesting) answers. The easiest way to do that is to sample the next predicted word from the SoftMax probability distribution. By combining that with the `loop_function` argument of `tf.nn.seq2seq.rnn_decoder`, it shouldn't be too difficult to add. After that, it should be possible to play with the SoftMax temperature to get more conservative or exotic predictions.
* Adding attention could potentially improve the predictions, especially for longer sentences. It should be straightforward by replacing `embedding_rnn_seq2seq` by `embedding_attention_seq2seq` on `model.py`.
* Having more data usually don't hurt. Training on a bigger corpus should be beneficial. [Reddit comments dataset](https://www.reddit.com/r/datasets/comments/59039y/updated_reddit_comment_dataset_up_to_201608/) seems the biggest for now (and is too big for this program to support it). Another trick to artificially increase the dataset size when creating the corpus could be to split the sentences of each training sample (ex: from the sample `Q:Sentence 1. Sentence 2. => A:Sentence X. Sentence Y.` we could generate 3 new samples: `Q:Sentence 1. Sentence 2. => A:Sentence X.`, `Q:Sentence 2. => A:Sentence X. Sentence Y.` and `Q:Sentence 2. => A:Sentence X.`. Warning: other combinations like `Q:Sentence 1. => A:Sentence X.` won't work because it would break the transition `2 => X` which links the question to the answer)
* The testing curve should really be monitored as done in my other [music generation](https://github.com/Conchylicultor/MusicGenerator) project. This would greatly help to see the impact of dropout on overfitting. For now it's just done empirically by manually checking the testing prediction at different training steps.
* For now, the questions are independent from each other. To link questions together, a straightforward way would be to feed all previous questions and answer to the encoder before giving the answer. Some caching could be done on the final encoder stated to avoid recomputing it each time. To improve the accuracy, the network should be retrain on entire dialogues instead of just individual QA. Also when feeding the previous dialogue to the encoder, new tokens `<Q>` and `<A>` could be added so the encoder knows when the interlocutor is changing. I'm not sure though that the simple seq2seq model would be sufficient to capture long term dependencies between sentences. Adding a bucket system to group similar input lengths together could greatly improve training speed.

## Credits

Credits for this code goes to [conchylucultor](https://github.com/Conchylicultor/DeepQA). I've merely created a wrapper to get people started. 
