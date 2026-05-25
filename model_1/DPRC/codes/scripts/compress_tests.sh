set -e
CUDA_VISIBLE_DEVICES=2 python test_compress.py --channel r --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
CUDA_VISIBLE_DEVICES=3 python test_compress.py --channel g --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
CUDA_VISIBLE_DEVICES=4 python test_compress.py --channel b --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
wait

exit 0

CUDA_VISIBLE_DEVICES=2 python test_compress.py --channel r --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
CUDA_VISIBLE_DEVICES=3 python test_compress.py --channel g --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
CUDA_VISIBLE_DEVICES=4 python test_compress.py --channel b --pixel_pitch 6.4 --prop_dist 20 --compress --dataset collected &
wait
