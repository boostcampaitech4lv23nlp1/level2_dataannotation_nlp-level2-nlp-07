import pickle
label_list = ['no_relation','stu:sub_study','stu:high_study','stu:alternate_names','stu:contributor',
            'stu:area','stu:research_group','stu:influence','stu:element','lan:high_language',
            'lan:sub_language','lan:product','lan:use_area','lan:alternate_names','lan:group_of_people']
print(len(label_list))
dict_label_to_num = {k : v for v,k in enumerate(label_list)}
# with open('dict_label_to_num.pkl', 'wb') as f:
#     pickle.dump(dict_label_to_num, f, pickle.HIGHEST_PROTOCOL)



# dict_num_to_label = {v : k for v,k in enumerate(label_list)}
# with open('dict_num_to_label.pkl', 'wb') as f:
#     pickle.dump(dict_num_to_label, f, pickle.HIGHEST_PROTOCOL)