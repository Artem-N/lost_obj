import os

# Define the mapping of classes
class_mapping = {
    0: 2,
    1: 0
}

def update_class_numbers(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            with open(input_filepath, "r") as file:
                lines = file.readlines()

            # Process each line in the file
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])

                # Replace class ID based on the mapping, if present
                new_class_id = class_mapping.get(class_id, class_id)  # default to class_id if not in mapping
                updated_line = f"{new_class_id} " + " ".join(parts[1:])
                updated_lines.append(updated_line)

            # Write the updated lines to a new file
            with open(output_filepath, "w") as file:
                for updated_line in updated_lines:
                    file.write(updated_line + "\n")

# Specify the input and output folders
input_folder = r"C:\Users\User\Desktop\dronevbird\valid\labels"
output_folder = r"C:\Users\User\Desktop\dronevbird\valid\labels"
update_class_numbers(input_folder, output_folder)

print("Class numbers updated successfully. New files are saved in the output folder.")
