import sys
import os
import json


def main():
    if len(sys.argv) < 3:
        print("Usage: %s <input_jsonl> <output_csv>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.exists(output_path):
        print('ERROR: Output path already exists')

    with open(output_path, 'wxb') as outfile:
        with open(input_path, 'rb') as infile:
            for line in infile:
                input = json.loads(line)
                id = input['id']
                pred_answer = input['pred_answer']
                outfile.write('%s,%s\n' % (id, pred_answer))


if __name__ == "__main__":
    # execute only if run as a script
    main()