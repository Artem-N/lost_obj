import os
import xml.etree.ElementTree as ET


def convert_bbox(size, bbox):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
    y_center = (bbox['ymin'] + bbox['ymax']) / 2.0
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax'] - bbox['ymin']
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return (x_center, y_center, width, height)


def convert_xml_to_yolo(xml_folder, output_folder, class_mapping):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_annotations = []

        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_mapping:
                continue
            cls_id = class_mapping[cls_name]
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': float(bndbox.find('xmin').text),
                'ymin': float(bndbox.find('ymin').text),
                'xmax': float(bndbox.find('xmax').text),
                'ymax': float(bndbox.find('ymax').text)
            }
            x_center, y_center, w, h = convert_bbox((width, height), bbox)
            yolo_annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write to .txt file
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        with open(os.path.join(output_folder, txt_filename), 'w') as txt_file:
            txt_file.write('\n'.join(yolo_annotations))


# Example usage:
# Define class mapping
class_mapping = {
    'person': 0,
    # Add other classes here
}

# Specify folders
xml_folder = r"C:\Users\User\Desktop\LLVIP\LLVIP\Annotations"  # Replace with your XML files directory
output_folder = r"C:\Users\User\Desktop\LLVIP\LLVIP\Annotations_YOLO"  # Replace with desired output directory

convert_xml_to_yolo(xml_folder, output_folder, class_mapping)
