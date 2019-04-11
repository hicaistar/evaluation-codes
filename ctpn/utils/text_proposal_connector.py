#coding:utf-8
import numpy as np
from text_connect_cfg import Config as TextLineCfg



class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self):
        self.matrix_builder=TextProposalMatrixBuilder()
        


    def group_text_proposals(self, text_proposals, scores, im_size):
        '''
            Group text proposals into list of text lines
            return:
                result: a list of different text lines, each item contains a list of text proposal indexes,
                which is at least adjacent to one another in the list
        '''
        matrix=self.matrix_builder.build_matrix(text_proposals, scores, im_size)
        result = matrix.sub_graphs_connected()
        return result

    def fit_y(self, X, Y, x1, x2):
        '''
            fit the X,Y into a line, calculate the corresponding y of x1,x2 
        '''
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        '''
            merge adjacent text_proposal into the textlines
            parameter:
                text_proposals: the bbox which satisfy the condition of text/non-text
                scores: the corresponding scores for each text_proposals
                im_size: the size the input image
            return:
                a list of text rectangle, which merge the text_proposal into the same group
        '''
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
        text_lines=np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]

            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score

        text_lines=clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            xmin,ymin,xmax,ymax=line[0],line[1],line[2],line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmin
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmax
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
    

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes

class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """
    def __init__(self):
        self.matrix_builder=TextProposalMatrixBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        '''
            Group text proposals into list of text lines
            return:
                result: a list of different text lines, each item contains a list of text proposal indexes,
                which is at least adjacent to one another in the list
        '''
        matrix=self.matrix_builder.build_matrix(text_proposals, scores, im_size)
        return matrix.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        '''
            fit the X,Y into a line, calculate the corresponding y of x1,x2 
        '''
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        '''
            merge adjacent text_proposal into the textlines
            parameter:
                text_proposals: the bbox which satisfy the condition of text/non-text
                scores: the corresponding scores for each text_proposals
                im_size: the size the input image
            return:
                a list of text quadrangle(represented by 4 vertexes), which merge the text_proposal into the same group
        '''
        # 这里就是尝试给出不同框所属的文本行,需要原图尺寸的信息，把不同的anchor分给不同的框
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)#首先还是建图，获取到文本行由哪几个小框构成
        
        text_lines=np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]#每个文本行的全部小框
            X = (text_line_boxes[:,0] + text_line_boxes[:,2]) / 2# 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:,1] + text_line_boxes[:,3]) / 2
            
            z1 = np.polyfit(X,Y,1)#多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0=np.min(text_line_boxes[:, 0])#文本行x坐标最小值
            x1=np.max(text_line_boxes[:, 2])#文本行x坐标最大值

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5#小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右(x坐标最小值最大值)对应的y坐标
            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右(x坐标最小值最大值)对应的y坐标
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            score=scores[list(tp_indices)].sum()/float(len(tp_indices))#求全部小框得分的均值作为文本行的均值

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)#文本行上端 线段 的y坐标的小值
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)#文本行下端 线段 的y坐标的大值
            text_lines[index, 4]=score#文本行得分
            text_lines[index, 5]=z1[0]#根据中心点拟合的直线的k，b，y = kx+b
            text_lines[index, 6]=z1[1]
            height = np.mean( (text_line_boxes[:,3]-text_line_boxes[:,1]) )#小框平均高度
            text_lines[index, 7]= height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
    
    
class TextProposalMatrixBuilder:
    """
        Build Text proposals into a  matrix to cluster each text_proposal box
    """
    ## 找相邻的连起来
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            ### 判断另一个box的x是不是在MAX_HORIZONTAL_GAP以内
            for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                ### adj_box_indices 表示在该x位置上的text_proposal的index的列表
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index): # 满足垂直方向的 IOU 条件
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

        
    def get_precursors(self, index):
        ''' return the precursors text_proposal of the index
        '''
        box=self.text_proposals[index]
        results=[]
        for left in range(int(box[0])-1, max(int(box[0]-TextLineCfg.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        ''' confirm the index is succession_index precursors
        '''
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        '''
            confirm the text proposals of index1,index2 meet the vertical condition of adjacent node
        '''
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM

    def build_matrix(self, text_proposals, scores, im_size):
        ''' build a matrix to indicate the adjacent relationship bewteen each text proposals
            parameters:
                text_proposals: the bbox which satisfy the condition of text/non-text
                scores: the corresponding scores for each text_proposals
                im_size: the size the input image
            return:
                matrix: a matrix store the adjacent relationship bewteen each text proposals
            ''' 
        self.text_proposals=text_proposals
        self.scores=scores
        self.im_size=im_size
        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1
        
        # boxes_table 维护的是一个长度为图片宽度的双重列表，每一个x的位置，会加上存在的text_proposa的index
        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        ## 维护一个所有text_proposal 对应的是不是相邻的text_proposal的matrix
        matrix=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            succession_index=successions[np.argmax(scores[successions])]   # 取相邻的text_proposal分值最高的那个
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                matrix[index, succession_index]=True
        return Adjacent_Matrix(matrix)

    



class Adjacent_Matrix:
    def __init__(self, matrix):
        ## 这里 graph 就是一个数组 np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool) 
        self.matrix=matrix

    def sub_graphs_connected(self):
        ''' 
        返回一个list,每个元素是一个list,list的每个元素都有一个相邻的index在list中
        '''
        sub_graphs=[]
        for index in range(self.matrix.shape[0]):
            '''如果 index的这列不是全为0,且index代表的行也不全为零,
               这里就是找一个联通域了
            ''' 
            if not self.matrix[:, index].any() and self.matrix[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.matrix[v, :].any():
                    v=np.where(self.matrix[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

