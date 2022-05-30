python3 dynamic_setup.py build_ext --inplace

for task in 'RightWhaleCalls' 'FruitFlies' 'MosquitoSound'
do
  python3 time_main.py --task $task --model CDIL
  python3 time_main.py --task $task --model TCN
  python3 time_main.py --task $task --model CNN
  python3 time_main.py --task $task --model Deformable
  python3 time_main.py --task $task --model GRU
  python3 time_main.py --task $task --model LSTM
  python3 time_main.py --task $task --model Transformer
  python3 time_main.py --task $task --model Linformer
  python3 time_main.py --task $task --model Performer
  python3 time_dynamic_main.py --task $task
done
