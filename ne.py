import numpy as np


class ClothesData:
    category = ""
    name = ""
    key_points = {}
    def __init__(self, name, category, kps):
        self.name = name
        self.category = category
        self.kp_enames = \
        "neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out".split(",")
        k = kps
        self.key_points = {}
        for i,v in enumerate(k):
            kp_name = self.kp_enames[i]
            v = np.array(v.split("_"), dtype=np.int32)
            self.key_points[kp_name] = v
    def get_norm(self):

        category = self.category
        
        #print(self.key_points)
        if category == "blouse" or \
                category == "dress" or\
                category == "outwear":
            x1,y1,t1 = self.key_points['armpit_right']
            x2,y2,t2 = self.key_points['armpit_left']
            #print("xyt:",x1,y1,t1)
            #print("xyt:",x2,y2,t2)
            #print("\n")
            if t1 == -1 or t2 == -1:
                return 512
            else:
               a = np.array([x1,y1]) 
               b = np.array([x2,y2]) 
               dis = np.sqrt(np.sum(np.square(a-b)))
               #print(x1,y1,x2,y2,t1,t2,dis)
               return dis
        else:
            x1,y1,t1 = self.key_points['waistband_right']
            x2,y2,t2 = self.key_points['waistband_left']
            if t1 == -1 or t2 == -1:
                return 512
            else:
               a = np.array([x1,y1]) 
               b = np.array([x2,y2]) 
               dis = np.sqrt(np.sum(np.square(a-b)))
               #print(x1,y1,x2,y2,t1,t2,dis)
               return dis
    def get_norm_dis(self, pred):
        norm = self.get_norm()
        count = 0
        res = []
        for i,kname in enumerate(self.kp_enames):
            x1,y1,t1 = self.key_points[kname]
            x2,y2,t2 = pred.key_points[kname]
            #print("xyt:",i,x1,y1,t1)
            #print("xyt:",x2,y2,t2)
            #print("\n")
            if t1 != 1: 
                res.append(0.0)
                continue
            if t2 != 1: 
                res.append(1.0)
                count += 1
                continue
            else:
               a = np.array([x1,y1]) 
               b = np.array([x2,y2]) 
               dis = np.sqrt(np.sum(np.square(a-b)))
               res.append(float(dis)/float(norm))
               count += 1

        #assert(count == 7)
        return np.sum(res), count

def get_total_ne(pred, label):
    res = []
    num = 0
    total = len(label)
    for i,name in enumerate(label):
        pred_data = pred[name]
        true_data = label[name]
        #print(name, true_data.key_points)
        ne,count = true_data.get_norm_dis(pred_data)
        res.append(ne)
        num += count
        if (i%100) == 0:
            print("<<=====FINISH:", (float(i+1.0)/(float(total)), "===>>>"))
            print(name,ne,count)
        #if num > 16:
        #    break

    res = np.sum(res)
    if num != 0:
        res = res / float(num)
    print("Final NE:", res)
    return res


def read_data(datafile):
    datas = dict()
    with open(datafile, "r") as f:
        firtline = f.readline()
        for image_info in f.readlines():
            items = image_info.replace("\n","").split(",")
            path, image_type = items[:2]
            image_name = path.split("/")[-1]

            kps=items[2:]
            datas[image_name] = \
            ClothesData(image_name,image_type,kps)

    return datas


def change_data_file(datafile,out):
    with open(datafile, "r") as f:
        firtline = f.readline()
        res = []
        for image_info in f.readlines():
            items = image_info.replace("\n","").split(",")
            path, image_type = items[:2]
            image_name = "Images/" + image_type + "/" + path
            items[0] = image_name
            res.append(",".join(items))
        res = "\n".join(res)
    with open(out,"w") as f:
        f.write(firtline)
        f.write(res)


        






if __name__ == '__main__':
#    change_data_file("result_pred.csv", "result.csv")
    true_datas = read_data("result_true.csv")
    pred_datas = read_data("result.csv")
    print(len(true_datas), len(pred_datas))
    get_total_ne(label=true_datas, pred=pred_datas)



