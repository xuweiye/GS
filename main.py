from config.Params import parser
from preprocess_data.patch_img import LevirCS_patch
from preprocess_data.patch_img_S import SPARCS_patch
from preprocess_data.patch_img_WHU import WHU_patch
from preprocess_data.patch_img_WFV import WFV_patch
from train import LevirCs_Train
from train_SPARCS import SPARCS_Train
from train_WHU import WHU_Train
from train_WFV import WFV_Train
from xiaorong_trian import xiaorong_SPARCS_Train
from Bound_train_WHU import Bou_WHU_Train
from bound_train_SPARCS import boudry_SPARCS_Train
from Bound_train_LevirCS import boudry_LevirCS_Train
from Bound_train_WFV import boundary_WFV_Train
from evalution import compute_mIoU,compute_mIoU_S
from deep_train_S import deep_SPARCS_Train
from deep_train_W import deep_WHU_Train
from deep_train_L import deep_LevirCS_Train
from deep_train_WFV import deep_WFV_Train
from RS_train_L import rs_LevirCS_Train
from RS_train_S import rs_SPARCS_Train
from RS_train_W import rs_WHU_Train
from RS_train_WFV import rs_WFV_Train
opt = parser.parse_args()

if opt.cut_img and opt.train_model == 0:
        if opt.dataset == 'LevirCS':
                LevirCS_patch(opt)
        if opt.dataset == 'SPARCS':
                SPARCS_patch(opt)
        if opt.dataset == 'WHU':
                WHU_patch(opt)
        if opt.dataset == 'WFV':
                WFV_patch(opt)

if opt.evalution and opt.train_model == 0:
        if opt.dataset == 'SPARCS':
                compute_mIoU_S(opt)
        if opt.dataset == 'LevirCS':
                compute_mIoU(opt)

if opt.train == 'train' and opt.train_model ==1:
        if opt.model == 'CAN':
                if opt.dataset == 'LevirCS':
                        train_code = LevirCs_Train(opt)
                        train_code.train_model()
                elif opt.dataset == 'SPARCS':
                        if opt.xiaorong:
                                train_code = xiaorong_SPARCS_Train(opt)
                                train_code.train_model()
                        else:
                                train_code = SPARCS_Train(opt)
                                train_code.train_model()
                elif opt.dataset == 'WHU':
                        train_code = WHU_Train(opt)
                        train_code.train_model()
                elif opt.dataset == 'WFV':
                        train_code = WFV_Train(opt)
                        train_code.train_model()

        if opt.model == 'Boundary':
                if opt.dataset == 'SPARCS':
                        train_code = boudry_SPARCS_Train(opt)
                        train_code.train_model()
                elif opt.dataset == 'WHU':
                        train_code = Bou_WHU_Train(opt)
                        train_code.train_model()
                elif opt.dataset == 'LevirCS':
                        train_code = boudry_LevirCS_Train(opt)
                        train_code.train_model()
                elif opt.dataset == 'WFV':
                        train_code = boundary_WFV_Train(opt)
                        train_code.train_model()
        if opt.model == 'deeplabv3':
                if opt.dataset == 'SPARCS':
                        train_code = deep_SPARCS_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'WHU':
                        train_code = deep_WHU_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'LevirCS':
                        train_code = deep_LevirCS_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'WFV':
                        train_code = deep_WFV_Train(opt)
                        train_code.train_model()
        if opt.model == 'RS':
                if opt.dataset == 'SPARCS':
                        train_code = rs_SPARCS_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'WHU':
                        train_code = rs_WHU_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'LevirCS':
                        train_code = rs_LevirCS_Train(opt)
                        train_code.train_model()
                if opt.dataset == 'WFV':
                        train_code = rs_WFV_Train(opt)
                        train_code.train_model()




