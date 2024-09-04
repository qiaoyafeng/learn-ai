import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

print(f"smile.feature_names: {smile.feature_names}")

with open("opensmile-feature_names.txt", "wt") as out_file:
    feature_names_str = "\n".join(smile.feature_names)
    out_file.write(feature_names_str)

y = smile.process_file('xu.wav')

print(f"y: {y}")


# y.to_csv("opensmile.csv")
