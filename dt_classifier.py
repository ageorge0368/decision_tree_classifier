import pandas as pd  # pandas library reads csv file https://pandas.pydata.org/docs/user_guide/io.html
import itertools  # for finding all combinations in a list (GINI INDEX)
import math


class Node:
    def __init__(self, attribute, parent_label):
        self.attribute = attribute  # splitting criterion
        self.children = []  # dictionary of children nodes
        self.parent_label = parent_label

    def addChild(self, child):  # dictionary is setup in {key: value} format
        self.children.append(child)  # add child to node.children

    def addFeature(self, feature):
        self.parent_label = feature

    def display(self, depth=0):  # prints fpTree in vertical format(if nodes line up vertically they have the same parent node)
        prefix = '  ' * depth
        print(f"{prefix}{self.attribute}{prefix}{self.parent_label}")
        for child in self.children:
            child.display(depth + 1)


def numOfClasses(Dataset, last_col):
    countList = []
    nameList = []
    for row in Dataset:
        row_index = Dataset.index(row)
        if Dataset[row_index][last_col] not in nameList:
            nameList.append(Dataset[row_index][last_col])
            countList.append(1)
        else:
            countList[nameList.index(Dataset[row_index][last_col])] += 1

    return countList, nameList

def majorityClasses(Dataset, last_col):
    temp_count, temp_name = numOfClasses(Dataset, last_col)
    maxCount = max(temp_count)
    return temp_name[temp_count.index(maxCount)]


def GenerateDecisionTree(Dataset, attribute_list, attribute_list_ref, selection_string, last_col):
    # create new node
    root = Node(None, None)
    # if (tuples in Dataset are of the same class)
    # return node as a leaf labeled as class c
    c = Dataset[0][last_col]  # literal class
    classList = []
    for row in Dataset:
        classList.append(row[last_col])  # class list

    if (all(value == c for value in classList)):
        root = Node(c, None)
        return root
    # if (attribute_list == {})
    if (len(attribute_list) == 0):
        # return node as a leaf labeled with the majority class in dataset
        root = Node(majorityClasses(Dataset, last_col), None)
        return root

    # attribute = Attribute_Selection_Method(Attribute_List, String)
    attribute, splitting_feature = Attribute_Selection_Method(Dataset, attribute_list, attribute_list_ref, selection_string, last_col)
    # label node with splitting criterion
    root = Node(attribute, None)
    # if (splitting criterion is categorical/discrete and math related, splitsa are allowed)
    # return attribute_list = attribute_list - {attribute}
    index = attribute_list_ref.index(attribute)
    attribute_list.remove(attribute)
    # for (each outcome j of the splitting criterion) {
    # Dj = set of tuples in D satisfying outcome j
    # if (Dj == NULL)
    # attach a leaf labeled with the majority class of  D
    # else attach the node returned by GenerateDecisionTree()
    # }

    # creating feature list
    if selection_string == "GINI":
        D_1 = []                            # meets condition           (yes)
        D_2 = []                            # doesn't meet condition    (no)

        for row in Dataset:
            row_index = Dataset.index(row)
            if Dataset[row_index][index] in splitting_feature:
                D_1.append(row)
            else:
                D_2.append(row)

       # check if all classes in D_1 are the same
        # make that a leaf node of that class
        c_D_1 = D_1[0][last_col]
        classList_D_1 = []
        for row in D_1:
            classList_D_1.append(row[last_col])  # class list

        if (all(value == c_D_1 for value in classList_D_1)):
            child_1 = Node(c_D_1, splitting_feature[0])
            root.addChild(child_1)
        else:
            child_1 = GenerateDecisionTree(D_1, attribute_list, attribute_list_ref, selection_string, last_col)
            child_1.addFeature(splitting_feature[0])  # probably wrong!
            root.addChild(child_1)

       # check if all classes in D_2 are the same
        # make that a leaf node of that class
        c_D_2 = D_2[0][last_col]
        classList_D_2 = []
        for row in D_2:
            classList_D_2.append(row[last_col])  # class list

        if (all(value == c_D_2 for value in classList_D_2)):
            child_2 = Node(c_D_2, splitting_feature[0])
            root.addChild(child_2)
        else:
            child_2 = GenerateDecisionTree(D_2, attribute_list, attribute_list_ref, selection_string, last_col)
            child_2.addFeature(splitting_feature[0])
            root.addChild(child_2)

        return root

    else:
        feature_list = []
        for row in Dataset:
            row_index = Dataset.index(row)
            if Dataset[row_index][index] not in feature_list:
                feature_list.append(Dataset[row_index][index])

        for feature in feature_list:
            D_j = []
            for row in Dataset:
                row_index = Dataset.index(row)
                if Dataset[row_index][index] == feature:
                    D_j.append(Dataset[row_index])  # creating D_j

            if len(D_j) == 0:
                child = Node(majorityClasses(Dataset, last_col), feature)
                root.addChild(child)
            else:
                child = GenerateDecisionTree(D_j, attribute_list, attribute_list_ref, selection_string, last_col)
                child.addFeature(feature)
                root.addChild(child)

        return root

def Attribute_Selection_Method(Dataset, attribute_list, attribute_list_ref, selection_string, last_col):
    selection = selection_string
    if selection == "GINI":
        temp_count, temp_name = numOfClasses(Dataset, last_col)
        sum = 0
        for class_name in temp_name:
            d_i = []
            for row in Dataset:
                if row[last_col] == class_name:
                    d_i.append([row])
            p_i = (len(d_i) / len(Dataset)) ** 2
            sum += p_i

        gini_d = 1 - sum

        gini_diff_val= []
        gini_diff_list = []

        gini_max_attr_diff = []

        unique_comb_list = []

        for a in attribute_list:
            singular_fea = []  # singular features for an attribute
            unique_comb = []  # unique combinations of features from an attribute

            for row in Dataset:  # singular_fee list creation
                row_index = Dataset.index(row)
                index = attribute_list_ref.index(a)
                if Dataset[row_index][index] not in singular_fea:  # column-row indexing
                    singular_fea.append(Dataset[row_index][index])

            num_of_combinations = 2 ** len(singular_fea) - 2
            if len(singular_fea) == 2:
                unique_comb.append(list(itertools.combinations(singular_fea, 1)))
            else:
                for i in range(1, (len(singular_fea))):  # unique_comb list creation (1 to n-1)
                    unique_comb.append(list(itertools.combinations(singular_fea, i)))

            unique_comb_list = []
            for j in unique_comb:
                for k in j:
                    unique_comb_list.append(list(k))  # unique_comb is list of lists

            for i in range(len(unique_comb_list)):
                yes_count = 0
                no_count = 0
                yes_list = []
                no_list = []

                for row in Dataset:
                    row_index = Dataset.index(row)
                    if Dataset[row_index][index] in unique_comb_list[i]:
                        yes_list.append(Dataset[row_index])  # D1
                        yes_count += 1
                    else:
                        no_list.append(Dataset[row_index])  # D2
                        no_count += 1

                # Gini(D1)
                temp_count_yes = []
                temp_name_yes = []
                temp_count_yes, temp_name_yes = numOfClasses(yes_list, last_col)
                sum_1 = 0
                for class_name in temp_name_yes:
                    D1 = []
                    for row in yes_list:
                        row_index = yes_list.index(row)
                        if row[last_col] == class_name:
                            D1.append([row])
                    p_i_1 = (len(D1) / len(yes_list)) ** 2
                    sum_1 += p_i_1

                gini_D1 = 1 - sum_1

                # Gini(D2)
                temp_count_no = []
                temp_name_no = []
                temp_count_no, temp_name_no = numOfClasses(no_list, last_col)
                sum_2 = 0
                for class_name in temp_name_no:
                    D2 = []
                    for row in no_list:
                        row_index = no_list.index(row)
                        if row[last_col] == class_name:
                            D2.append([row])
                    p_i_2 = (len(D2) / len(no_list)) ** 2
                    sum_2 += p_i_2

                gini_D2 = 1 - sum_2

                # Gini(a)
                gini_a = ((yes_count / len(Dataset)) * gini_D1) + ((no_count / len(Dataset)) * gini_D2)

                gini_diff_list.append([gini_d - gini_a, unique_comb_list[i]])
                gini_diff_val.append(gini_d - gini_a)

            max_gini_diff = max(gini_diff_val)
            gini_max_attr_diff.append(gini_diff_list[gini_diff_val.index(max_gini_diff)])

            gini_diff_list = []
            gini_diff_val = []

            index += 1

        max_attr_diff = max(gini_max_attr_diff)

        return attribute_list[gini_max_attr_diff.index(max_attr_diff)], max_attr_diff[1]

    elif selection == "INFORMATION GAIN":
        temp_count, temp_name = numOfClasses(Dataset, last_col)
        sum = 0
        for class_name in temp_name:
            d_i = []
            for row in Dataset:
                row_index = Dataset.index(row)
                if row[last_col] == class_name:
                    d_i.append([row])
            p_i = (len(d_i) / len(Dataset))
            addend = p_i * math.log(p_i, 2)
            sum += addend

        info_d = sum * -1

        gain_diff_list = []

        gain_max_attr_diff = []

        for a in attribute_list:
            singular_fea = []  # singular features for an attribute
            info_a_i = 0

            for row in Dataset:
                row_index = Dataset.index(row)  # singular_fee list creation
                index = attribute_list_ref.index(a)
                if Dataset[row_index][index] not in singular_fea:  # column-row indexing
                    singular_fea.append(Dataset[row_index][index])

            for feature in singular_fea:
                d_i_j_count = 0
                d_i_j = []

                for row in Dataset:
                    row_index = Dataset.index(row)
                    if Dataset[row_index][index] == feature:  # d_i_j
                        d_i_j_count += 1
                        d_i_j.append(row)

                # Info(D_ij)
                temp_count_j, temp_name_j = numOfClasses(d_i_j, last_col)
                sum_j = 0
                for class_name in temp_name_j:
                    d_i_j_yes = []
                    for row in d_i_j:
                        row_index = d_i_j.index(row)
                        if row[last_col] == class_name:
                            d_i_j_yes.append(row)
                    p_i_j = (len(d_i_j_yes) / len(d_i_j))
                    addend_j = p_i_j * math.log(p_i_j, 2)
                    sum_j += addend_j

                info_d_i_j = sum_j * -1

                info_a_i += ((d_i_j_count / len(Dataset)) * info_d_i_j)

            gain_a_i = info_d - info_a_i
            gain_diff_list.append(gain_a_i)

        max_gain_attr_diff = max(gain_diff_list)

        return attribute_list[gain_diff_list.index(max_gain_attr_diff)], "No Splitting Feature (IG)"

    else:  # GAIN RATIO
        temp_count, temp_name = numOfClasses(Dataset, last_col)
        sum = 0
        for class_name in temp_name:
            d_i = []
            for row in Dataset:
                row_index = Dataset.index(row)
                if row[last_col] == class_name:
                    d_i.append([row])
            p_i = (len(d_i) / len(Dataset))
            addend = p_i * math.log(p_i, 2)
            sum += addend

        info_d = sum * -1

        gain_ratio_list = []

        for a in attribute_list:
            singular_fea = []  # singular features for an attribute
            info_a_i = 0

            for row in Dataset:
                row_index = Dataset.index(row)  # singular_fee list creation
                index = attribute_list_ref.index(a)
                if Dataset[row_index][index] not in singular_fea:  # column-row indexing
                    singular_fea.append(Dataset[row_index][index])

            split_info = 0

            for feature in singular_fea:
                d_i_j_count = 0
                d_i_j = []

                feature_count = 0
                for row in Dataset:
                    row_index = Dataset.index(row)
                    if Dataset[row_index][index] == feature:  # d_i_j
                        d_i_j_count += 1
                        d_i_j.append(row)
                        feature_count += 1

                # Info(D_ij)
                temp_count_j, temp_name_j = numOfClasses(d_i_j, last_col)
                sum_j = 0
                for class_name in temp_name_j:
                    d_i_j_yes = []
                    for row in d_i_j:
                        row_index = d_i_j.index(row)
                        if row[last_col] == class_name:
                            d_i_j_yes.append(row)
                    p_i_j = (len(d_i_j_yes) / len(d_i_j))
                    addend_j = p_i_j * math.log(p_i_j, 2)
                    sum_j += addend_j

                info_d_i_j = sum_j * -1

                info_a_i += ((d_i_j_count / len(Dataset)) * info_d_i_j)

                feature_addend = ((feature_count / len(Dataset)) * math.log((feature_count / len(Dataset)), 2)) * -1

                split_info += feature_addend

            gain_a_i = info_d - info_a_i

            gain_ratio_a = (gain_a_i / split_info)
            gain_ratio_list.append(gain_ratio_a)

        max_gain_ratio = max(gain_ratio_list)

        return attribute_list[gain_ratio_list.index(max_gain_ratio)], "No Splitting Feature (GR)"

def test_tree_g(dataset, attribute_list_ref, root, last_col):
  col_index = attribute_list_ref.index(root.attribute)
  accuracy = 0
  #print(attribute_list_ref)
  for row in dataset:
    ptr = root
    row_index = dataset.index(row)
    while(ptr.children != []):
        if ptr.children[0].parent_label == dataset[row_index][col_index]:
            if ptr.children[0].attribute not in attribute_list_ref:
                ptr = ptr.children[0]
                break
            else:
                col_index = attribute_list_ref.index(ptr.children[0].attribute)
            ptr = ptr.children[0]
        else:
            if ptr.children[1].attribute not in attribute_list_ref:
                ptr = ptr.children[1]
                break
            else:
                col_index = attribute_list_ref.index(ptr.children[1].attribute)
            ptr = ptr.children[1]

    if dataset[row_index][last_col] == ptr.attribute:
        accuracy += 1

  accuracy = accuracy / len(dataset)
  return accuracy

def test_tree(dataset, attribute_list_ref, root, last_col):
  accuracy = 0
  #print(attribute_list_ref)
  for row in dataset:
    col_index = attribute_list_ref.index(root.attribute)
    ptr = root
    row_index = dataset.index(row)
    while_flag = True
    i = 0
    while(1):
        for i in range(len(ptr.children)):
            if ptr.children[i].parent_label == dataset[row_index][col_index]:           # you have to go to child node
                if ptr.children[i].attribute not in attribute_list_ref:                 # leaf node
                    while_flag = False
                    # flag to break out of while loop
                    break
                else:                                                                    # another child nodeS
                    col_index = attribute_list_ref.index(ptr.children[i].attribute)
                break
                # flag to break out of for loop

        if ptr.children == []:
            if while_flag == False:
                break

        ptr = ptr.children[i]

    if dataset[row_index][last_col] == ptr.attribute:
        accuracy += 1

  accuracy = accuracy / len(dataset)
  return accuracy

def main():
    Dataframe = pd.read_csv(r"C:\Users\ageor\Downloads\lymphography_fixed.csv") # read data into pandas dataframe from csv file
    Dataframe.astype(str)
    # Dataframe2 = Dataframe[Dataframe.columns[1:]] #remove first column of labels
    Dataset = Dataframe.values.tolist() #convert to list of lists
    d1 = Dataframe.sample(frac = 0.7) #train
    d2 = Dataframe.drop(d1.index) #test
    dataset1 = d1.values.tolist()
    dataset2 = d2.values.tolist()

    last_col = 18  # last col for classes (dependent on each dataset)
    index = 0  # index for each attribute

    attribute_list =    ["lymphatics", "block_of_affere", "bl._of_lymph._c", "bl._of_lymph._s",
                         "by_pass", "extravasates", "regeneration_of", "early_uptake_in",
                         "lymphatics", "lym.nodes_dimin", "lym.nodes_enlar", "changes_in_lym.",
                         "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms",
                         "dislocation_of", "exclusion_of_no", "no._of_nodes_in"]

    attribute_list_ref = ["lymphatics", "block_of_affere", "bl._of_lymph._c", "bl._of_lymph._s",
                         "by_pass", "extravasates", "regeneration_of", "early_uptake_in",
                         "lymphatics", "lym.nodes_dimin", "lym.nodes_enlar", "changes_in_lym.",
                         "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms",
                         "dislocation_of", "exclusion_of_no", "no._of_nodes_in"]
    #attribute_list = ["age", "income", "student", "credit_rating"]
    #attribute_list_ref = ["age", "income", "student", "credit_rating"]
    #Dataset = [["<=30", "high", "no", "fair", "no"],
   #            ["<=30", "high", "no", "excellent", "no"],
    #           ["31...40", "high", "no", "fair", "yes"],
     #          [">40", "medium", "no", "fair", "yes"],
      #         [">40", "low", "yes", "fair", "yes"],
       #        [">40", "low", "yes", "excellent", "no"],
       #        ["31...40", "low", "yes", "excellent", "yes"],
        #       ["<=30", "medium", "no", "fair", "no"],
         #      ["<=30", "low", "yes", "fair", "yes"],
          #     [">40", "medium", "yes", "fair", "yes"],
           #    ["<=30", "medium", "yes", "excellent", "yes"],
            #   ["31...40", "medium", "no", "excellent", "yes"],
         #      ["31...40", "high", "yes", "fair", "yes"],
          #     [">40", "medium", "no", "excellent", "no"]]

    root1 = GenerateDecisionTree(dataset1, attribute_list, attribute_list_ref, "GINI", last_col)
    print("\nAttribute or Leaf     Meets Condition? -> Condition")
    root1.display()
    print("\n")

    attribute_list = ["lymphatics", "block_of_affere", "bl._of_lymph._c", "bl._of_lymph._s",
                         "by_pass", "extravasates", "regeneration_of", "early_uptake_in",
                         "lymphatics", "lym.nodes_dimin", "lym.nodes_enlar", "changes_in_lym.",
                         "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms",
                         "dislocation_of", "exclusion_of_no", "no._of_nodes_in"]
    root2 = GenerateDecisionTree(dataset1, attribute_list, attribute_list_ref, "INFORMATION GAIN", last_col)
    print("\nAttribute or Leaf     Feature")
    root2.display()
    print("\n")

    attribute_list = ["lymphatics", "block_of_affere", "bl._of_lymph._c", "bl._of_lymph._s",
                         "by_pass", "extravasates", "regeneration_of", "early_uptake_in",
                         "lymphatics", "lym.nodes_dimin", "lym.nodes_enlar", "changes_in_lym.",
                         "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms",
                         "dislocation_of", "exclusion_of_no", "no._of_nodes_in"]
    root3 = GenerateDecisionTree(dataset1, attribute_list, attribute_list_ref, "GAIN RATIO", last_col)
    print("\nAttribute or Leaf     Feature")
    root3.display()
    print("\n")

    accuracy1 = test_tree_g(dataset2, attribute_list_ref, root1, last_col)
    print("Gini Index Accuracy:\t" + str(accuracy1))

    accuracy2 = test_tree(dataset2, attribute_list_ref, root2, last_col)
    print("Information Gain Accuracy:\t" + str(accuracy2))

    accuracy3 = test_tree(dataset2, attribute_list_ref, root3, last_col)
    print("Gain Ratio Accuracy:\t" + str(accuracy3))

if __name__ == "__main__":
    main()