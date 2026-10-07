# autoedit: BASED IRL монтажа за DaVinci Resolve

Десетте „skills“ од пакетот *BASED IRL editing* (увоз, груб рез, камера-агли, фин рез, титлови, производи, производи
зад личноста, B-roll, штембил и музика, аудио) префрлени од Premiere Pro на DaVinci Resolve, како алатка од командна
линија.

Алатката ги гледа суровите снимки, одлучува што да се сече и гради по една Resolve временска линија за секое кратко
видео. Ти ја отвораш и го дотеруваш со око, како што кажуваат SOP-ите.

> **Важно.** Сè што може да се тестира без Resolve е тестирано. Но **не е пуштено врз вистински DaVinci Resolve**, а
> Whisper, лицата и Claude делот се тестирани со замени, зашто тие модели и клучеви ги нема таму каде што е напишано.
> Прв пат пушти на пробен проект.

## Прв обид, чекор по чекор

**1. Програми.** Ти требаат Python 3.12 и ffmpeg.

- Mac (со [Homebrew](https://brew.sh)): `brew install python@3.12 ffmpeg`
- Windows: `winget install Python.Python.3.12` и `winget install Gyan.FFmpeg`, па затвори го терминалот и отвори го пак.

**2. Кодот.** На GitHub отвори го репото `zafirovdine358-dev/Kostadin-Zafirov`, префрли се на гранката
`claude/laughing-goodall-adi4f8`, па *Code > Download ZIP* и отпакувај. Отвори терминал (Mac: Terminal, Windows:
Command Prompt) и влези во отпакуваната папка: напиши `cd ` и повлечи ја папката во прозорецот.

**3. Инсталација** (само еднаш):

```
python3 -m venv .venv                  # Windows: py -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[asr,faces,images]"
```

Во секој нов терминал пушти ја втората линија пак. За „производи зад личноста“ подоцна додај `mediapipe`:
`pip install -e ".[matte]"`.

**4. Проверка.** `autoedit doctor` печати PASS, WARN или FAIL за секоја работа. FAIL мора да се поправи; WARN значи
дека некоја функција нема да работи (на пр. без Whisper нема титлови и фин рез, без модел за лица нема камера-агли).

**5. Поставки** (еднаш, од истата папка):

```
autoedit init --based-root "~/Documents/CLIENT WORK/BASED" --work "~/Documents/BASED Auto Edit"
```

Ова го пишува `autoedit.json` во тековната папка, па понатаму командите пушти ги од таа папка.

**Моделите.** Ако си го правел поставувањето за Premiere, тие веќе се во `~/Documents/Claude Tools` (`whisper-base/`,
`models/face_detection_yunet_2023mar.onnx`, `models/face_recognition_sface_2021dec.onnx`): `init` ги наоѓа сам и
печати `found whisper: ...`. Инаку додај `--whisper`, `--yunet` и `--sface` со патеките. Папката за Whisper мора да ги
има сите четири датотеки: `model.bin` (големата, околу 150 MB), `config.json`, `tokenizer.json` и `vocabulary.txt`.
`autoedit doctor` кажува која недостасува.

**6. Пробај на една снимка, без Resolve:**

```
autoedit run --dry-run --folder "/патека/до/29-09-26 Lian" --shorts G1291
```

`--folder` е папката со снимките од еден ден (повлечи ја во терминалот за да се залепи патеката), а `--shorts` е
името на еден клип без наставката (`G1291` за `G1291.MP4`). Оригиналите само се читаат: сè што се создава оди во
работната папка. Првпат Whisper се симнува сам (треба интернет). На крај се печати табела и патека до `report.md`:
отвори го и види што е сечено и што треба да провериш.

**7. Во Resolve Studio.** Направи нов празен проект за проба. *DaVinci Resolve > Preferences > System > General >
External scripting using: Local*, Save (ако не се поврзе, рестартирај го Resolve). Потоа во терминалот:
`autoedit apply`. Во *Media Pool* се појавува папка `BATCH 1`, а во *Timelines* линија `BATCH 1 - G1291`. Погледни
еден кадар: главата на говорникот треба да е на средина, со очите на околу 30% од врвот. Pan/Tilt/Zoom е делот што
најмногу бара проверка.

**Без Studio:** `autoedit apply --no-resolve` ги пишува `G1291.edl` и `G1291.srt` во работната папка. Во Resolve:
*File > Import > Timeline* (EDL) и *File > Import > Subtitle* (SRT). Добиваш само резови и титлови, без кадрирање,
анимации и исчистен глас. Од Resolve 21.1 Python скриптирањето е само за Studio, па Free не може да ги изгради
временските линии сам. Кое издание имаш: *DaVinci Resolve > About DaVinci Resolve*.

Сè друго е по избор (види табела во [README.md](README.md)): без `pillow` нема производи, без `mediapipe` нема „производи
зад личноста“.

## Користење

```
autoedit run --dry-run        # најновата снимка -> план (edit.json) + report.md, Resolve не се допира
autoedit status               # табела по кратко видео
autoedit apply                # гради временски линии во Resolve
autoedit apply --no-resolve   # само EDL + SRT датотеки (за рачен увоз)
```

Или чекор по чекор, како во пакетот: `autoedit ingest`, `rough-cut`, `camera-angles`, `fine-cut`, `subtitles`,
`products-behind`, `broll --product "curl cream"`, `stamps-and-music --stamp WM-XXXX.png`, `audio-fix`.
Ако пуштиш чекор повторно, сè што доаѓа по него се ресетира.

- **Resolve Studio:** *Preferences > System > General > External scripting using: Local*, па `autoedit apply`.
- **Resolve Free до 21.0:** еднаш `autoedit install-resolve-script`, па во Resolve *Workspace > Scripts > BASED Auto
  Edit*.
- **Resolve Free 21.1 и понов:** `autoedit apply --no-resolve` и рачен увоз на EDL и SRT.

Ако Resolve не може да се достигне, `apply` сам ги запишува тие две датотеки и завршува со код 3.

Штембилот го симнуваш сам од Editor Portal (9:16 PNG). Алатката никогаш не го отвора порталот и не бара, не чува и не
запишува линк до него.

## Што добиваш во Resolve

По една временска линија на кратко видео (`BATCH n - G1291`), 1080x1920 на 29.97. Постоечка временска линија со исто
име никогаш не се заменува (добиваш `v2`). V1 = снимката, V2/V3 = производ зад личноста, V4 = производи, V5 = B-roll,
V6 = штембил, V7 = BASED графика, A1 = исчистен глас, A3 = музика, плус трака за титлови и маркери (зелен
`STABILISED`, црвен `BAD ANGLE - fix`, жолт `CHECK showing?`).

## Ограничувања

- Не е тестирано врз вистински Resolve. `apply` ја чита секоја временска линија назад и пишува каде не се совпаѓа со
  планот: прочитај го тоа.
- Никој не може да слуша: аудиото е мерено (ниво, врв, длабочина на паузи), не слушано. Твоите уши одлучуваат.
- Resolve API нема клучни рамки, ефекти, маски ни транзиции, затоа движењата на камерата се неколку мирни кадри, а
  анимациите (производи, штембил) и чистењето на гласот се испечени со ffmpeg.
- Прво пробај на пробен проект и погледни еден кадар: Pan/Tilt/Zoom конвенциите се делот што треба да го провериш.
