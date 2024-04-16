import numpy as np 
try:
	from rdkit import Chem 
	from rdkit.Chem import AllChem
except:
	pass 


def plot_hist(prefix_name, prediction, label):
	import seaborn as sns
	import matplotlib.pyplot as plt
	figure_name = prefix_name + "_histogram.png"
	positive_prediction = [prediction[i] for i in range(len(label)) if label[i]==1]
	negative_prediction = [prediction[i] for i in range(len(label)) if label[i]==0]
	sns.distplot(positive_prediction, hist=True,  kde=False, bins=50, color = 'blue', label = 'positive')
	sns.distplot(negative_prediction, hist=True,  kde=False, bins=50, color = 'red', label = 'negative')
	plt.legend()
	plt.savefig(figure_name)
	return 

def replace_strange_symbol(text):
	for i in "[]'\n/":
		text = text.replace(i,'_')
	return text

#  xml read blog:  https://blog.csdn.net/yiluochenwu/article/details/23515923 
def walkData(root_node, prefix, result_list):
	temp_list =[prefix + '/' + root_node.tag, root_node.text]
	result_list.append(temp_list)
	# children_node = root_node.getchildren()
	children_node = list(root_node)
	if len(children_node) == 0:
		return
	for child in children_node:
		walkData(child, prefix = prefix + '/' + root_node.tag, result_list = result_list)


def dynamic_programming(s1, s2):
	arr2d = [[0 for i in s2] for j in s1]
	if s1[0] == s2[0]:
		arr2d[0][0] = 1
	for i in range(1, len(s1)):
		if s1[i]==s2[0]:
			arr2d[i][0] = 1
		else:
			arr2d[i][0] = arr2d[i-1][0] 
	for i in range(1,len(s2)):
		if s2[i]==s1[0]:
			arr2d[0][i] = 1 
		else:
			arr2d[0][i] = arr2d[0][i-1]
	for i in range(1,len(s1)):
		for j in range(1,len(s2)):
			if s1[i] == s2[j]:
				arr2d[i][j] = arr2d[i-1][j-1] + 1 
			else:
				arr2d[i][j] = max(arr2d[i-1][j], arr2d[i][j-1])
	return arr2d[len(s1)-1][len(s2)-1]


def get_path_of_all_xml_file():
	input_file = "./data/trials/all_xml.txt"
	with open(input_file, 'r') as fin:
		lines = fin.readlines()
	input_file_lst = [i.strip() for i in lines]
	return input_file_lst 


def remove_multiple_space(text):
	text = ' '.join(text.split())
	return text 

def nctid_2_xml_file_path(nctid):
	assert len(nctid)==11
	prefix = nctid[:7] + "xxxx"
	datafolder = os.path.join("./ClinicalTrialGov/", prefix, nctid+".xml")
	return datafolder 


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp

def smiles2fp(smiles):
	try:
		mol = Chem.MolFromSmiles(smiles)
		fp = fingerprints_from_mol(mol)
		return fp 
	except:
		return np.zeros((1, 2048), np.int32)

def smiles_lst2fp(smiles_lst):
	fp_lst = [smiles2fp(smiles) for smiles in smiles_lst]
	fp_mat = np.concatenate(fp_lst, 0)
	fp = np.mean(fp_mat,0)
	return fp	


# if __name__ == "__main__":
# 	input_file_lst = get_path_of_all_xml_file() 
# 	print(input_file_lst[:5])
# '''
# input_file_lst = [ 
# 	'ClinicalTrialGov/NCT0000xxxx/NCT00000102.xml', 
#  	'ClinicalTrialGov/NCT0000xxxx/NCT00000104.xml', 
#  	'ClinicalTrialGov/NCT0000xxxx/NCT00000105.xml', 
# 	  ... ]
# '''

