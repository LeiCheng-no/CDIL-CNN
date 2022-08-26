for task in 'image' 'text_4000' 'pathfinder32' 'retrieval_4000'
do
  python3 lra_main.py --task $task --model CDIL
  python3 lra_main.py --task $task --model TCN
  python3 lra_main.py --task $task --model CNN
  python3 lra_main.py --task $task --model Deformable
  python3 lra_main.py --task $task --model GRU
  python3 lra_main.py --task $task --model LSTM
done
