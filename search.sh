for bs in 32 64 128 256 512
do
  for lr in 1e-2 1e-4 1e-6 1e-8
  do
    for wd in 0 1e-2
    do
      CHECKPOINT="koc-finetuning-bs${bs}-lr${lr}-wd${wd}"
      accelerate launch --config_file accelerate/default_config.yaml train.py \
        ${CHECKPOINT} \
        --batch_size ${bs} \
        --learning_rate ${lr} \
        --weight_decay ${wd}
      CHECKPOINT="${CHECKPOINT}-adapter"
      accelerate launch --config_file accelerate/default_config.yaml train.py \
        ${CHECKPOINT} \
        --batch_size ${bs} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --add_adapter
      CHECKPOINT="koc-finetuning-bs${bs}-lr${lr}-wd${wd}-deepspeed"
      accelerate launch --config_file accelerate/deepspeed_config.yaml train.py \
        ${CHECKPOINT} \
        --batch_size ${bs} \
        --learning_rate ${lr} \
        --weight_decay ${wd}
      for psl in 16 128 512 1024
      do
        for pp in "True" "False"
        do
          CHECKPOINT="koc-ptuning-bs${bs}-lr${lr}-wd${wd}-psl${psl}"
          if [ pp -eq "True" ] ; then
            for phs in 128 512 1024 2048
            do
              CHECKPOINT="${CHECKPOINT}-phs${phs}"
              accelerate launch --config_file accelerate/default_config.yaml train.py \
                ${CHECKPOINT} \
                --batch_size ${bs} \
                --learning_rate ${lr} \
                --weight_decay ${wd} \
                --pre_seq_len ${psl} \
                --prefix_projection ${pp} \
                --prefix_hidden_size ${phs} \
                --p_tuning
            done
          else
            accelerate launch --config_file accelerate/default_config.yaml train.py \
              ${CHECKPOINT} \
              --batch_size ${bs} \
              --learning_rate ${lr} \
              --weight_decay ${wd} \
              --pre_seq_len ${psl} \
              --prefix_projection ${pp} \
              --p_tuning
          fi
        done
      done
    done
  done
done

