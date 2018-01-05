from numpy.random import RandomState
# import imageio

def test_segmentation():
    PRNG = RandomState()
    PRNG2 = RandomState()
    if args.seed > 0:
        PRNG.seed(args.seed)
        PRNG2.seed(args.seed)

    transform = Compose([
        [ColorJitter(prob=0.75), None],
        Merge(),
        Expand((0.8, 1.5)),
        RandomCompose([
            # RandomResize(1, 1.5),
            RandomRotate(10),
            RandomShift(0.1)]),
        Scale(300),
        # ElasticTransform(100),
        RandomCrop(300),
        HorizontalFlip(),
        Split([0, 3], [3, 6]),
        #[SubtractMean(mean=VOC.MEAN), None],
        ], 
        PRNG, 
        border='constant', 
        fillval=VOC.MEAN,
        anchor_index=3)

    voc_dataset = VOCSegmentation(
                        root=args.root,
                        image_set=[('2007', 'trainval')],
                        transform=transform,
                        instance=False)
    viz = Viz()
    
    results = []
    count = 0
    i = PRNG2.choice(len(voc_dataset))
    for _ in range(1000):
        img, target = voc_dataset[i]
        img2 = viz.blend_segmentation(img, target)

        con = np.hstack([img, target, img2])
        results.append(con)
        cv2.imshow('result', con[..., ::-1])
        c = cv2.waitKey(500)

        if c == 27 or c == ord('q'):   # ESC / 'q'
            break
        elif c == ord('c') or count >= 3:
            count = 0
            i = PRNG2.choice(len(voc_dataset))
        count += 1

    # imageio.mimsave('mask.gif', results, duration=0.5)


if __name__ == '__main__':
    from transforms import *
    from pascal_voc import VOC, VOCSegmentation, Viz

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='voc dataset root path', default='path/to/your/VOCdevkit')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    test_segmentation()