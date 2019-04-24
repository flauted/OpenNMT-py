from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


if __name__ == "__main__":
    pred = open("../pred.txt")

    import pickle
    import os

    YT = "/data/sd0/here/yt2t_2/youtube2text_iccv15"

    with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
        ann = pickle.load(f, encoding="latin-1")

    vid2anns = {}
    for vid_name, data in ann.items():
        for d in data:
            try:
                vid2anns[vid_name].append(d["tokenized"])
            except KeyError:
                vid2anns[vid_name] = [d["tokenized"]]

    test_files = open("/data/sd0/here/yt2t_2/yt2t_test_files.txt")

    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge(),
        "Cider": Cider(),
        "Spice": Spice()
    }

    gts = {}
    res = {}
    for outp, filename in zip(pred, test_files):
        filename = filename.strip("\n")
        outp = outp.strip("\n")
        vid_id = os.path.splitext(filename)[0]
        anns = vid2anns[vid_id]
        gts[vid_id] = anns
        res[vid_id] = [outp]

    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    print(scores)
