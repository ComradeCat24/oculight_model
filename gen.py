import sys
import replicate

output = replicate.run(
    "rmokady/clip_prefix_caption:9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8",
    input={"image": open(sys.argv[1], "rb")}
)

print(output)
