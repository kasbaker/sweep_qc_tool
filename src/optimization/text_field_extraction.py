from pathlib import Path

base_path = Path(__file__).parent
text_path = base_path.joinpath('v_clamp_series_comments.txt')

# print(base_path)
# print(text_path)

with open(str(text_path)) as file:
    comment_str = file.read()

# comment_str.splitlines()
# print(comment_str)
# separator = "HS#0:"
# print(comment_str.split(separator))
# print(comment_str)

# key = "V-Clamp Holding Enable: "
# key = "V-Clamp Holding Enable: "

# _, key_str, value_str = comment_str.partition(key)
# if value_str:
#     value_str = value_str.partition("\n")[0]
# print(key_str)
# print(value_str)
#
# key2 = "V-Clamp Holding Level: "
#
# _, key_str, value_str = comment_str.partition(key2)
# if value_str:
#     value_str = value_str.partition("\n")[0]
# print(key_str)
# print(value_str)

# print(part_str[1])
# print(part_str[2])
# print(value_str[0])


# filtered_str = comment_str.trim

amplifier_settings_keys = (
    "V-Clamp Holding Enable: ", "V-Clamp Holding Level: ",
    "RsComp Enable: ", "RsComp Correction: ", "RsComp Prediction: ",
    "Whole Cell Comp Enable: ", "Whole Cell Comp Cap: ", "Whole Cell Comp Resist: "
)

# amplifier_settings_keys = None

settings_dict = dict.fromkeys(amplifier_settings_keys)

for key in amplifier_settings_keys:
    value_str = comment_str.partition(key)[2]
    if value_str:
        value = value_str.splitlines()[0]
        # value = value_str.partition("\n")[0]
        settings_dict[key] = value

for key, value in settings_dict.items():
    print(key)
    print(value)
    print("----------")
# print(settings_dict)
