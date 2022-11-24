model_name="koc"
checkpoint_prefix="checkpoint/BEST_"

pretrained_model="EleutherAI/polyglot-ko-3.8b"

for checkpoint_path in ${checkpoint_prefix}${model_name}*
do
  echo ${checkpoint_path}
  checkpoint="${checkpoint_path#"${checkpoint_prefix}"}"
  echo ${checkpoint}
  mt="${checkpoint##*mt}"
  mt="${mt%%--*}"
  echo ${mt}

  if [ ${mt} = "finetuning" ] ; then
    python test.py --saved_model ${checkpoint} \
      --pretrained_model ${pretrained_model}
  elif [ ${mt} = "adapter" ] ; then
    python test.py --saved_model ${checkpoint} --add_adapter \
      --pretrained_model ${pretrained_model}
  elif [ ${mt} = "ptuning" ] ; then
    psl="${checkpoint##*psl}"
    psl="${psl%%--*}"
    pp="${checkpoint##*pp}"
    pp="${pp%%--*}"
    if [ ${pp} == "True" ] ; then
      phs="${checkpoint##*phs}"
      phs="${phs%%--*}"
      python test.py --saved_model ${checkpoint} --p_tuning \
        --pretrained_model ${pretrained_model} \
        --pre_seq_len ${psl} \
        --prefix_projection ${pp} \
        --prefix_hidden_size ${phs}
    else
      python test.py --saved_model ${checkpoint} --p_tuning \
        --pretrained_model ${pretrained_model} \
        --pre_seq_len ${psl} \
        --prefix_projection ${pp}
    fi
  fi
done
