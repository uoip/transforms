from numpy.random import RandomState
# import imageio

def test_bboxes():
    PRNG = RandomState()
    PRNG2 = RandomState()
    if args.seed > 0:
        PRNG.seed(args.seed)
        PRNG2.seed(args.seed)

    transform = Compose([           
            [ColorJitter(prob=0.5)],   # or write [ColorJitter(), None]
            BoxesToCoords(),
            HorizontalFlip(),
            Expand((1, 4), prob=0.5),
            ObjectRandomCrop(),
            Resize(300),
            CoordsToBoxes(),
            #[SubtractMean(mean=VOC.MEAN)],
            ], 
            PRNG, 
            mode=None, 
            fillval=VOC.MEAN, 
            outside_points='clamp')

    viz = Viz()
    voc_dataset = VOCDetection(
                        root=args.root, 
                        image_set=[('2007', 'trainval')],
                        keep_difficult=True,
                        transform=transform)

    results = []
    count = 0
    i = PRNG2.choice(len(voc_dataset))
    for _ in range(100):
        img, boxes, labels = voc_dataset[i]
        if len(labels) == 0:
            continue

        img = viz.draw_bbox(img, boxes, labels, True)
        results.append(img)
        cv2.imshow('0', img[:, :, ::-1])
        c = cv2.waitKey(500)
        if c == 27 or c == ord('q'):   # ESC / 'q'
            break
        elif c == ord('c') or count >= 5:
            count = 0
            i = PRNG2.choice(len(voc_dataset))
        count += 1

    # imageio.mimsave('bboxes.gif', results, duration=0.5)


if __name__ == '__main__':
    from transforms import *
    from pascal_voc import VOC, VOCDetection, Viz

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='voc dataset root path', default='path/to/your/VOCdevkit')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    test_bboxes()