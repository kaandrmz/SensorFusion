import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
import scipy.special
import scipy.io
import torch

from utils.brisque.brisque.brisque import BRISQUE

# Add import for CLIP-IQA
try:
    from torchmetrics.multimodal import CLIPImageQualityAssessment
except ImportError:
    raise ImportError("Please install torchmetrics with 'pip install torchmetrics[multimodal]' to use CLIP_IQA")

# Add import for pyiqa
try:
    import pyiqa
except ImportError:
    raise ImportError("Please install pyiqa")

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF 
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        data_range = 255.0
        return ssim(image_F, image_A, data_range=data_range) + ssim(image_F, image_B, data_range=data_range)

    @classmethod
    def NIQE(cls, image_F):
        cls.input_check(image_F)
        # Use the image directly - assume grayscale conversion is already handled
        return cls._compute_niqe(image_F)
        
    @classmethod
    def CLIP_IQA(cls, image_F):
        """
        Calculate CLIP Image Quality Assessment score.
        CLIP-IQA uses the CLIP model trained on (image, text) pairs to evaluate image quality.
        
        Args:
            image_F: Fused image (grayscale or RGB)
            
        Returns:
            float: Quality score between 0 and 1 (higher is better)
        """
        
        # Convert to torch tensor format for the model
        # The model expects tensor of shape [N, C, H, W] with values normalized to [0, 1]
        img_tensor = torch.from_numpy(image_F.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
        
        try:
            # Only use image quality prompt by default
            metric = CLIPImageQualityAssessment(data_range=1.0)
            
            # Move to the same device as the input tensor if it's on CUDA
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            metric = metric.to(device)
            img_tensor = img_tensor.to(device)
            
            # Compute the CLIP-IQA score
            with torch.no_grad():
                score = metric(img_tensor)
                
                # If multiple prompts are used, extract the quality score
                if isinstance(score, dict):
                    score = score['quality']
                
                # Return the score as a float (average if batch size > 1)
                return score.mean().item()
        except Exception as e:
            print(f"Error computing CLIP-IQA: {e}")
            return float('nan')
        
    @classmethod
    def MUSIQ(cls, image_F):
        """
        Calculate MUSIQ (Multi-scale Image Quality Transformer) score.
        MUSIQ is a no-reference image quality assessment method that uses transformer architecture.
        
        Args:
            image_F: Fused image (RGB format)
            
        Returns:
            float: Quality score (higher is better)
        """
        try:
            # Create MUSIQ metric with default settings
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            musiq_metric = pyiqa.create_metric('musiq', device=device)
            
            # Check if the image is already a tensor
            if isinstance(image_F, torch.Tensor):
                img_tensor = image_F
            else:
                # Convert numpy array to tensor if needed
                # The image should be in RGB format with shape [H, W, C]
                if len(image_F.shape) != 3:
                    # Convert grayscale to RGB if needed
                    raise ValueError("MUSIQ requires RGB images")
                
                # Convert to tensor of shape [C, H, W] normalized to [0, 1]
                img_tensor = torch.from_numpy(image_F.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
                img_tensor = img_tensor.to(device)
            
            # Compute the MUSIQ score
            with torch.no_grad():
                score = musiq_metric(img_tensor)
                
                # Return the score as a float
                return score.item() if isinstance(score, torch.Tensor) else score
        except Exception as e:
            print(f"Error computing MUSIQ: {e}")
            return float('nan')
        
    @classmethod
    def _compute_niqe(cls, gray_image):
        gamma_range = np.arange(0.2, 10, 0.001)
        a = scipy.special.gamma(2.0/gamma_range)
        a *= a
        b = scipy.special.gamma(1.0/gamma_range)
        c = scipy.special.gamma(3.0/gamma_range)
        prec_gammas = a/(b*c)
        
        def aggd_features(imdata):
            #flatten imdata
            imdata.shape = (len(imdata.flat),)
            imdata2 = imdata*imdata
            left_data = imdata2[imdata<0]
            right_data = imdata2[imdata>=0]
            left_mean_sqrt = 0
            right_mean_sqrt = 0
            if len(left_data) > 0:
                left_mean_sqrt = np.sqrt(np.average(left_data))
            if len(right_data) > 0:
                right_mean_sqrt = np.sqrt(np.average(right_data))

            if right_mean_sqrt != 0:
                gamma_hat = left_mean_sqrt/right_mean_sqrt
            else:
                gamma_hat = np.inf
            #solve r-hat norm

            imdata2_mean = np.mean(imdata2)
            if imdata2_mean != 0:
                r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
            else:
                r_hat = np.inf
            rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

            #solve alpha by guessing values that minimize ro
            pos = np.argmin((prec_gammas - rhat_norm)**2)
            alpha = gamma_range[pos]

            gam1 = scipy.special.gamma(1.0/alpha)
            gam2 = scipy.special.gamma(2.0/alpha)
            gam3 = scipy.special.gamma(3.0/alpha)

            aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
            bl = aggdratio * left_mean_sqrt
            br = aggdratio * right_mean_sqrt

            #mean parameter
            N = (br - bl)*(gam2 / gam1)
            return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

        def ggd_features(imdata):
            nr_gam = 1/prec_gammas
            sigma_sq = np.var(imdata)
            E = np.mean(np.abs(imdata))
            rho = sigma_sq/E**2
            pos = np.argmin(np.abs(nr_gam - rho))
            return gamma_range[pos], sigma_sq

        def paired_product(new_im):
            shift1 = np.roll(new_im.copy(), 1, axis=1)
            shift2 = np.roll(new_im.copy(), 1, axis=0)
            shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
            shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

            H_img = shift1 * new_im
            V_img = shift2 * new_im
            D1_img = shift3 * new_im
            D2_img = shift4 * new_im

            return (H_img, V_img, D1_img, D2_img)

        def gen_gauss_window(lw, sigma):
            sd = np.float32(sigma)
            lw = int(lw)
            weights = [0.0] * (2 * lw + 1)
            weights[lw] = 1.0
            sum = 1.0
            sd *= sd
            for ii in range(1, lw + 1):
                tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
                weights[lw + ii] = tmp
                weights[lw - ii] = tmp
                sum += 2.0 * tmp
            for ii in range(2 * lw + 1):
                weights[ii] /= sum
            return weights

        def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
            if avg_window is None:
                avg_window = gen_gauss_window(3, 7.0/6.0)
            assert len(np.shape(image)) == 2
            h, w = np.shape(image)
            mu_image = np.zeros((h, w), dtype=np.float32)
            var_image = np.zeros((h, w), dtype=np.float32)
            image = np.array(image).astype('float32')
            scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
            scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
            scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
            scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
            var_image = np.sqrt(np.abs(var_image - mu_image**2))
            return (image - mu_image)/(var_image + C), var_image, mu_image

        def _niqe_extract_subband_feats(mscncoefs):
            alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
            pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
            alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
            alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
            alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
            alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
            return np.array([alpha_m, (bl+br)/2.0,
                    alpha1, N1, bl1, br1,  # (V)
                    alpha2, N2, bl2, br2,  # (H)
                    alpha3, N3, bl3, bl3,  # (D1)
                    alpha4, N4, bl4, bl4,  # (D2)
            ])

        def extract_on_patches(img, patch_size):
            h, w = img.shape
            patch_size = np.int_(patch_size)
            patches = []
            for j in range(0, h-patch_size+1, patch_size):
                for i in range(0, w-patch_size+1, patch_size):
                    patch = img[j:j+patch_size, i:i+patch_size]
                    patches.append(patch)

            patches = np.array(patches)
            
            patch_features = []
            for p in patches:
                patch_features.append(_niqe_extract_subband_feats(p))
            patch_features = np.array(patch_features)

            return patch_features

        def _get_patches_generic(img, patch_size, is_train, stride):
            h, w = np.shape(img)
            if h < patch_size or w < patch_size:
                raise ValueError("Input image is too small")

            # ensure that the patch divides evenly into img
            hoffset = (h % patch_size)
            woffset = (w % patch_size)

            if hoffset > 0: 
                img = img[:-hoffset, :]
            if woffset > 0:
                img = img[:, :-woffset]

            img = img.astype(np.float32)
            # Use cv2 resize instead of scipy.misc.imresize which is deprecated
            img2 = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)

            mscn1, var, mu = compute_image_mscn_transform(img)
            mscn1 = mscn1.astype(np.float32)

            mscn2, _, _ = compute_image_mscn_transform(img2)
            mscn2 = mscn2.astype(np.float32)

            feats_lvl1 = extract_on_patches(mscn1, patch_size)
            feats_lvl2 = extract_on_patches(mscn2, patch_size//2)

            feats = np.hstack((feats_lvl1, feats_lvl2))

            return feats

        def get_patches_test_features(img, patch_size, stride=8):
            return _get_patches_generic(img, patch_size, 0, stride)

        def niqe_core(inputImgData):
            patch_size = 96
            # Load parameters from file
            try:
                # Try to load from file - assumes data directory exists in the same directory as this file
                import os
                module_path = os.path.dirname(os.path.abspath(__file__))
                params = scipy.io.loadmat(os.path.join(module_path, 'data', 'niqe_image_params.mat'))
                pop_mu = np.ravel(params["pop_mu"])
                pop_cov = params["pop_cov"]
            except Exception as e:
                # Instead of using fallback values, raise an error
                raise FileNotFoundError(f"Could not load NIQE model parameters: {str(e)}. Please ensure the file 'data/niqe_image_params.mat' exists in the same directory as Evaluator.py")

            M, N = inputImgData.shape

            # Check if image size is sufficient
            if M <= (patch_size*2+1) or N <= (patch_size*2+1):
                # Return a high score for small images that can't be processed
                return 100.0

            feats = get_patches_test_features(inputImgData, patch_size)
            sample_mu = np.mean(feats, axis=0)
            sample_cov = np.cov(feats.T)

            X = sample_mu - pop_mu
            covmat = ((pop_cov+sample_cov)/2.0)
            pinvmat = np.linalg.pinv(covmat)
            niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

            return niqe_score
            
        return niqe_core(gray_image)

    @classmethod
    def BRISQUE(cls, image_F):
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
        BRISQUE is a no-reference image quality assessment method.
        
        Args:
            image_F: Fused image (grayscale or RGB)
            
        Returns:
            float: Quality score (lower is better, typically in range [0, 100])
        """
        try:
            # Initialize BRISQUE metric
            brisque = BRISQUE()
            
            # Convert image to uint8 if needed
            if image_F.dtype != np.uint8:
                image_F = (image_F * 255).astype(np.uint8)
            
            # Calculate BRISQUE score 
            score = brisque.score(image_F)
            return score
            
        except Exception as e:
            print(f"Error computing BRISQUE: {e}")
            return float('nan')


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB
