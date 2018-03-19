import random
def shuffle_annotation(fin, fout):
    with open(fin, "r") as f:
        firtline = f.readline().replace("\n","")
        res = []
        for image_info in f.readlines():
            res.append(image_info.replace("\n",""))
        random.shuffle(res)
    with open(fout, "w") as f:
        f.write(firtline+"\n")
        f.write("\n".join(res))

if __name__ == "__main__":
    shuffle_annotation("./train.csv", "./train_shuffle.csv")
        
