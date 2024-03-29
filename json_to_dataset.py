# coding=gbk
import argparse
import base64
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils


def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    count = os.listdir(json_file)
    print(count)
    for i in range(0, len(count)):
        if count[i][-5:]!=".json":
            continue
        file_path = os.path.join(json_file, count[i])
        if os.path.isfile(file_path):
            if args.out is None:
                out_dir = osp.basename(file_path).replace('.', '_')
                out_dir = osp.join(osp.dirname(file_path), out_dir)
            else:
                # out_dir = args.out#批量生成指定输出目录
                out_dir = osp.basename(file_path).replace('.', '_')
                out_dir = osp.join(osp.dirname(args.out), out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            data = json.load(open(file_path))

            # 禁用json图片数据
            if False and data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(file_path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {'_background_': 0}
            for shape in sorted(data['shapes'], key=lambda x: x['label']):
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            label_names = [None] * (max(label_name_to_value.values()) + 1)
            for name, value in label_name_to_value.items():
                label_names[value] = name
            lbl_viz = utils.draw_label(lbl, img, label_names)
            
            # 不保存原图
            # PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
