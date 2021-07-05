# # 更换替换字符为空格
# path=/home/xuwd/projects/DuEE-pytorch/submit/new2/ner-span.json
# sed -i "s/®/ /g"  ${path}
# # 将argument中顿号分隔，e.g.{"role": "高管职位", "argument": "董事会秘书、财务负责人"}==>分成两个
# path2=/home/xuwd/projects/DuEE-pytorch/submit/new2/ner-span-split.json
# python3 split_dunhao.py --originalpath ${path} --splitpath ${path2}
# # 将不同eventtype合并成一个，不同的argument只出现一次
# writepath=/home/xuwd/projects/DuEE-pytorch/submit/new2/last-ner.json
# python3 merge_zy2.py --path2 ${path2} --writepath ${writepath}





# #更换替换字符为空格
path22=/home/xuwd/projects/DuEE-pytorch/submit/new2/test2.all_cpt_all_ids.json
sed -i "s/®/ /g"  ${path22}

#将argument中顿号分隔，e.g.{"role": "高管职位", "argument": "董事会秘书、财务负责人"}==>分成两个
path2=/home/xuwd/projects/DuEE-pytorch/submit/new2/test2.all_cpt_all_ids-split.json
python3 split_dunhao.py --originalpath ${path22} --splitpath ${path2}

#将不同eventtype合并成一个，不同的argument只出现一次
writepath=/home/xuwd/projects/DuEE-pytorch/submit/new2/test2zy.json
python3 merge_zy2.py --path2 ${path2} --writepath ${writepath}

# addidpath=/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin-39zy56.53-addid.json
# python3 add_id_zyto3w.py --lessidpath ${writepath} --addidpath ${addidpath}


#直接将两份结果拼在一起，不去重，之后通过merge_zy2去重
xupath=/home/xuwd/projects/DuEE-pytorch/submit/new2/last-ner.json
writetmppath=/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin-tmp.json
python3 merge_zy.py --xupath ${xupath} --zypath ${addidpath} --writepath ${writetmppath}

#去重
writepath3=/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin.json
python3 merge_zy2.py --path2 ${writetmppath} --writepath ${writepath3}




