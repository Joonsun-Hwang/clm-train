checkpoint_name="koc"

pretrained_model="EleutherAI/polyglot-ko-3.8b"
method_list="finetuning lora ptuning"

bs_list="32 128 512"
wm_list="0 100"
lr_list="1e-2 1e-4 1e-6 1e-8"
wd_list="0 1e-1"

psl_list="4 16 128 1024"
pp_list="True False"
phs_list="128 512 1024 2048"

dist_list="default deepspeed"

for mt in ${method_list} ; do
  for bs in ${bs_list} ; do
    for wm in ${wm_list} ; do
      for lr in ${lr_list} ; do
        for wd in ${wd_list} ; do
          for dist in ${dist_list} ; do
            checkpoint_tmp="${model_name}--mt${mt}--dist${dist}--bs${bs}--wm${wm}--lr${lr}--wd${wd}"
            if [ ${mt} = "finetuning" ] ; then
              checkpoint="${checkpoint_tmp}"
              accelerate launch --config_file "accelerate/${dist}_config.yaml" train.py
                ${checkpoint}
                --pretrained_model ${pretrained_model}
                --batch_size ${bs}
                --num_warmup_steps ${wm}
                --learning_rate ${lr}
                --weight_decay ${wd}
            elif [ ${mt} = "lora" ] && [ ${dist} = "default" ] ; then
              checkpoint="${checkpoint_tmp}"
              accelerate launch --config_file "accelerate/${dist}_config.yaml" train.py
                ${checkpoint}
                --pretrained_model ${pretrained_model}
                --batch_size ${bs}
                --num_warmup_steps ${wm}
                --learning_rate ${lr}
                --weight_decay ${wd}
                --add_adapter
            elif [ ${mt} = "ptuning" ] && [ ${dist} = "default" ] ; then
              for psl in ${psl_list} ; do
                for pp in ${pp_list} ; do
                  if [ ${pp} = "True" ] ; then
                    for phs in ${phs_list} ; do
                      checkpoint="${checkpoint_tmp}--psl${psl}--pp${pp}--phs${phs}"
                      accelerate launch --config_file "accelerate/${dist}_config.yaml" train.py
                        ${checkpoint}
                        --pretrained_model ${pretrained_model}
                        --batch_size ${bs}
                        --num_warmup_steps ${wm}
                        --learning_rate ${lr}
                        --weight_decay ${wd}
                        --pre_seq_len ${psl}
                        --prefix_projection ${pp}
                        --prefix_hidden_size ${phs}
                        --p_tuning
                    done
                  else
                    checkpoint="${checkpoint_tmp}--psl${psl}--pp${pp}"
                    accelerate launch --config_file "accelerate/${dist}_config.yaml" train.py
                      ${checkpoint}
                      --pretrained_model ${pretrained_model}
                      --batch_size ${bs}
                      --num_warmup_steps ${wm}
                      --learning_rate ${lr}
                      --pre_seq_len ${psl}
                      --prefix_projection ${pp}
                      --p_tuning
                  fi
                done
              done
            fi
          done
        done
      done
    done
  done
done

