for bit_length in 20 30 40 50 60 70 80 90 100;
do
  for noisy_label in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5;
  do
    qsub train_cpu.sh $bit_length $noisy_label
  done
done

