from autoedit import captions as cp, config

C = config.merge(config.DEFAULTS, {})["captions"]


def words(spec, spk="H", t0=0.0, step=0.35):
    out, t = [], t0
    for w in spec.split():
        out.append(cp.TWord(w, t, t + step - 0.05, spk))
        t += step
    return out


def texts(ws, cuts=()):
    caps, _ = cp.build(ws, list(cuts), C)
    return [x.text for x in caps]


def test_based_gets_its_own_caption_and_questions_get_a_mark():
    assert texts(words("Do you follow base on TikTok?")) == ["do you follow", "based", "on tiktok?"]
    assert texts(words("Are you following BASED?")) == ["are you following", "based"]      # no ? after based


def test_at_most_three_words_and_four_splits_two_two():
    out = texts(words("this is super good stuff"))
    assert all(len(t.split()) <= 3 for t in out)
    assert texts(words("curly hair needs water daily."))[0].startswith("curly hair")   # a name is never split
    assert texts(words("it works well today")) == ["it works", "well today"]      # 4 -> 2 + 2


def test_no_caption_ends_on_a_weak_word_and_groups_stay_whole():
    out = texts(words("I use the sea salt spray every day"))
    assert out == ["I use the", "sea salt spray", "every day"]       # 'the' + a 3-word name does not fit in 3 words
    assert texts(words("try the texture powder tonight")) == ["try", "the texture powder", "tonight"]
    assert texts(words("I love the curl cream so much")) == ["I love", "the curl cream", "so much"]
    # no weak ending whenever it can be avoided
    assert texts(words("we use the best spray for your hair every day")) == \
        ["we use", "the best spray", "for your hair", "every day"]
    assert texts(words("you should definitely try the leave-in conditioner tonight"))[2] == "the leave-in conditioner"
    assert texts(words("it is a really good product"))[-1] == "good product"


def test_i_forms_and_fillers_and_repeats():
    assert texts(words("um I think yeah yeah okay")) == ["I think yeah x2", "okay"] or texts(words("um I think yeah yeah okay"))[0].startswith("I think")
    assert texts(words("yeah yeah"))[0] == "yeah x2"
    assert texts(words("I'm sure I've seen it"))[0].startswith("I'm sure")


def test_long_captions_are_broken_up():
    assert texts(words("definitely recommend")) == ["definitely", "recommend"]
    assert texts(words("that you're following")) == ["that you're", "following"]
    assert texts(words("sea salt spray")) == ["sea salt spray"]


def test_asr_repairs():
    assert texts(words("try coke cream today")) == ["try curl cream", "today"] or "curl cream" in " ".join(texts(words("try coke cream today")))
    assert "skin revival" in " ".join(texts(words("the skin rib eye spray")))
    assert "leave-in conditioner" in " ".join(texts(words("the leaving conditioner")))
    assert "based" in texts(words("following face on tiktok")) or "based" in " ".join(texts(words("following face on tiktok")))
    assert "face" in " ".join(texts(words("my face looks tired")))       # a real face stays a face


def test_other_brands_are_replaced_and_muted():
    caps, mutes = cp.build(words("camila rose is good"), [], C)
    assert [x.text for x in caps][0] == "*other brand*" and caps[0].kind == "brand"
    assert len(mutes) == 1 and mutes[0][0] == 0.0 and mutes[0][1] > 0.4


def test_phrase_breaks_on_speaker_change_punctuation_gap_and_cuts():
    ws = words("hello there", "H") + words("hi", "G", t0=1.0)
    assert texts(ws) == ["hello there", "hi"]
    ws = words("one two") + words("three four", t0=0.7 + 0.1)
    assert texts(ws, cuts=[0.75]) == ["one two", "three four"]
    ws = words("so yes. okay then", step=0.3)
    assert texts(ws) == ["so yes", "okay then"]


def test_retime_is_back_to_back_frame_exact_and_min_dur():
    ws = words("one two three four five six seven eight nine", step=0.12)
    caps, _ = cp.build(ws, [], C)
    out = cp.retime(caps, 29.97, C, 5.0)
    for a, b in zip(out, out[1:]):
        assert abs(a.e - b.s) < 1e-9
    assert all(x.e - x.s >= 0.19 for x in out)
    assert all(abs(x.s * 30000 / 1001 - round(x.s * 30000 / 1001)) < 1e-6 for x in out)
    assert out[-1].e <= 5.0 + 1e-6


def test_srt_format(tmp_path):
    caps = [cp.Caption(0.0, 1.5, "hello there"), cp.Caption(1.5, 2.2, "based", kind="based")]
    p = cp.write_srt(caps, str(tmp_path / "a.srt"), skip_kinds=("based",))
    assert open(p).read() == "1\n00:00:00,000 --> 00:00:01,500\nhello there\n\n"
