import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='extracting arc string from ENAS stdout')
parser.add_argument('--stdout', metavar='STDOUT', default='', 
                    help='path to ENAS stdout')
parser.add_argument('--arc-length', default=8, type=int,
					help='fixed length of architectures(default: 8)')

def extract_arc_string(args):
	if not os.path.exists(args.stdout):
		raise ValueError("path to stdout doesn't exist")

	arc_string_output_file = open("arc_string_output.txt", "w")
	val_acc_list = []
	test_acc_list = []
	break_count = 0
	arc_length = int(args.arc_length) - 1
	with open(args.stdout) as f:
		for line in f:
			if line[0] == '[' and line[-2]==']':
				break_count += 1
				arc_string_output_file.write(line[1:-2]+'\n')
				print line[1:-2]
				if break_count != 0 and break_count % arc_length == 0:
					# pass
					# arc_string_output_file.write('\n')
					print '\n'
			if line[:16] == 'valid_accuracy: ':
				print float(line[16:-1])
				val_acc_list.append(float(line[16:-1]))
			if line[:15] == 'test_accuracy: ':
				print float(line[15:-1])
				test_acc_list.append(float(line[15:-1]))

	np.save('valid_accuracy_array.npy', np.array(val_acc_list))
	np.save('test_accuracy_array.npy', np.array(test_acc_list))
	print(len(val_acc_list), len(test_acc_list))
	print('bv and bv_test')
	print(max(val_acc_list))
	for i in range(len(val_acc_list)):
		if max(val_acc_list) == val_acc_list[i]:
			print test_acc_list[i]

if __name__ == "__main__":
	args = parser.parse_args()
	extract_arc_string(args) 