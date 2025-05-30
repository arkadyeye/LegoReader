import cv2
import numpy as np

'''

this class it totally made by gpt. I have no idea what is going here
it just do the work :)

'''

class IconFinder:
    def __init__(self, large_image, icon_image, threshold=0.8, scales=None):
        self.large_image = large_image
        self.icon_image = icon_image
        self.threshold = threshold
        self.scales = scales or [1.0]
        self.large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
        self.icon_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
        self.all_matches = []
        self.filtered_matches = []

    def find_all_matches(self):
        """Collect all raw matches above the threshold from all scales."""
        h, w = self.icon_gray.shape
        self.all_matches.clear()

        for scale in self.scales:
            resized_icon = cv2.resize(self.icon_gray, (int(w * scale), int(h * scale)))
            result = cv2.matchTemplate(self.large_gray, resized_icon, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= self.threshold)

            for pt in zip(*loc[::-1]):
                x, y = pt
                box_w, box_h = resized_icon.shape[::-1]
                score = result[y, x]
                self.all_matches.append({
                    'box': (x, y, x + box_w, y + box_h),
                    'score': score,
                    'scale': scale
                })

        # Sort by descending match score
        self.all_matches.sort(key=lambda m: -m['score'])

    def filter_matches(self, overlap_thresh=0.3):
        """Run NMS on all_matches and keep best non-overlapping ones."""
        if not self.all_matches:
            return []

        boxes = np.array([m['box'] for m in self.all_matches])
        scores = np.array([m['score'] for m in self.all_matches])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(scores)

        pick = []
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            suppress = [last]

            for pos in range(len(idxs) - 1):
                i = idxs[pos]
                xx1 = max(x1[last], x1[i])
                yy1 = max(y1[last], y1[i])
                xx2 = min(x2[last], x2[i])
                yy2 = min(y2[last], y2[i])

                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                overlap = float(w * h) / areas[i]

                if overlap > overlap_thresh:
                    suppress.append(i)

            suppress_idx = [np.where(idxs == s)[0][0] for s in suppress if s in idxs]
            idxs = np.delete(idxs, suppress_idx)

        self.filtered_matches = [self.all_matches[i] for i in pick]
        return self.filtered_matches

    def get_filtered_boxes(self):
        """
        Return list of bounding boxes from filtered matches as standard Python ints.
        Format: [(x1, y1, x2, y2), ...]
        """
        return [
            tuple(int(v) for v in match['box'])
            for match in self.filtered_matches
        ]

    def draw_matches(self, use_filtered=True):
        """Draw filtered or raw matches."""
        image = self.large_image.copy()
        matches = self.filtered_matches if use_filtered else self.all_matches

        for m in matches:
            x1, y1, x2, y2 = m['box']
            score = m['score']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return image
