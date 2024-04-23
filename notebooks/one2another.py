# conda create -n env_cv2xfeatures python=3.6
# conda activate env_cv2xfeatures
# pip install opencv-python==3.4.2.17
# pip install opencv-contrib-python==3.4.2.17


import cv2
import numpy as np

default_coord_finder_opts = {
    'N_FEATURES': 100,
    'MIN_MATCH_COUNT': 10,
    'TREES_COUNT': 5,
    'CHECKS': 30,
    'KNN_DIFF': 0.7,
    'use_cap_weap_blur': False,
    'use_cap_helm_blur': False,
    'cap_weap_blur': [5, 5],
    'cap_helm_blur': [5, 5]
}


class coord_finder(object):
    def __init__(self, opts=None):
        if not opts:
            opts = dict(default_coord_finder_opts)
        self._opts = None
        self._sift = None
        self._flann = None
        self.set_opts(opts)

    def set_opts(self, opts):
        self._opts = opts
        self._sift = cv2.xfeatures2d.SURF_create()#(opts['N_FEATURES'])
        FLANN_INDEX_KDTREE = 0
        trees_count = opts['TREES_COUNT']
        checks_count = opts['CHECKS']
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees_count)
        search_params = dict(checks=checks_count)
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

    def get_coords(self, img_helm, img_weap, weap_center=None,
                   return_matches_img=True, weap_mask=None, helm_mask=None):
        """
        метод считает координаты точки weap_center кадра img_weap в системе координат img_helm
        :param img_helm: кадр с камеры шлема
        :param img_weap: кадр с камеры оружия
        :param weap_center: точка в СК камеры оружия, которую нужно найти в СК камеры шлема
        :param return_matches_img: возвращать ли кадр подробностей
        :param weap_mask: маска для кадра с камеры оружия
        :param helm_mask: маска для кадра с камеры шлема
        :return: (ret, x, y, [matches_img])
                ret - True/False (успех/ не успех)
                x,y - координаты точки weap_center в СК img_helm
                matches_img - картинка с подробностями (если return_matches_img=True)
        """
        try:
            # if self._opts['use_cap_weap_blur']:
            #     img_weap = cv2.blur(img_weap, tuple(self._opts['cap_weap_blur']))
            # if self._opts['use_cap_helm_blur']:
            #     img_helm = cv2.blur(img_helm, tuple(self._opts['cap_helm_blur']))

            kp1, des1 = self._sift.detectAndCompute(img_helm, helm_mask)
            kp2, des2 = self._sift.detectAndCompute(img_weap, weap_mask)

            matches = self._flann.knnMatch(des1, des2, 2)

            good = []
            for m, n in matches:
                if m.distance < self._opts['KNN_DIFF'] * n.distance:
                    good.append(m)
            img2 = None
            x, y = None, None
            success = False
            if len(good) > self._opts['MIN_MATCH_COUNT']:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w, _ = img_helm.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                if weap_center is None:
                    weap_center = [w / 2, h / 2]
                center_p = np.float32(weap_center).reshape(-1, 1, 2)
                center_p = cv2.perspectiveTransform(center_p, M)[0, 0, :]
                x, y = center_p[0], center_p[1]
                if return_matches_img:
                    img2 = cv2.circle(img_weap, (center_p[0], center_p[1]), 5, (255, 0, 0), 3)
                    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    img_helm = cv2.circle(img_helm, (int(weap_center[0]), int(weap_center[1])), 5, (255, 0, 0), 3)

                success = True
            else:
                print("Not enough matches are found - %d/%d" % (len(good), self._opts['MIN_MATCH_COUNT']))
                matchesMask = None
            if return_matches_img:
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                img3 = cv2.drawMatches(img_helm, kp1, img2, kp2, good, None, **draw_params)
                return (success, x, y, img3)
            return (success, x, y)
        except Exception as e:
            print(f"Error in get_coords:  {e}")
            return (False, None, None, None) if return_matches_img else (False, None, None)


def main():
    # cap2 = cv2.VideoCapture(1)
    cap1 = cv2.VideoCapture(0)
    cf = coord_finder()

    while True:
        ret1, img_helm = cap1.read()
        # ret2, img_weap = cap2.read()

        # ret, x, y, img3 = cf.get_coords(img_helm, img_weap)
        # if ret:
        #     cv2.imshow('sum', img3)
        cv2.imshow('sum', img_helm)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap1.release()
    # cap2.release()
    cv2.destroyAllWindows()

def main1():
    img1 = cv2.imread('pics/1.jpg')
    img2 = cv2.imread('pics/2.jpg')
    cf = coord_finder()
    ret, x, y, img3 = cf.get_coords(img2, img1)
    while True:

        if ret:
            cv2.imshow('sum', img3)
        # cv2.imshow('sum', img_helm)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main1()
