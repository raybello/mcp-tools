from moviepy import VideoFileClip, TextClip, ImageClip, CompositeVideoClip, clips_array, vfx
from moviepy.video.tools.subtitles import SubtitlesClip
        

length = 10
filename = "assets/output/download.mp4"
# font_file = "assets/font/Glass-mE4m.ttf"
# font_file = "assets/font/SF-Pro.ttf"
font_file = "Arial"
image_file = "assets/images/test_image.png"
image_file = "assets/images/erasebg-transformed.png"
output_file = "assets/output/final.mp4"

clip1 = VideoFileClip(filename).with_effects([vfx.Margin(10)]).subclipped(0, length)
clip2 = VideoFileClip(filename).subclipped(0, length)

combined = clips_array([[clip1], [clip2]])
combined.write_videofile("assets/output/example.mp4")

# ##########################################################

# We load all the clips we want to compose
background = VideoFileClip(filename).subclipped(0, 9)
title = TextClip(
    font_file,
    text="Big Buck Bunny",
    bg_color="black",
    font_size=30,
    color="blue",
    text_align="center",
    duration=5,
    # method="caption",
)
author = TextClip(
    font_file,
    text="Blender Foundation",
    # bg_color="black",
    font_size=30,
    color="blue",
    text_align="center",
    duration=5,
    # method="caption",
)
copyright = TextClip(
    font_file,
    text="© CC BY 3.0",
    # bg_color="black",
    font_size=30,
    color="blue",
    text_align="center",
    duration=5,
    # method="caption"
)
logo = ImageClip(image_file, duration=8).resized(height=200)

# We want our title to be at the center horizontaly and start at 25%
# of the video verticaly. We can set as "center", "left", "right",
# "top" and "bottom", and % relative from the clip size
title = title.with_position(("center", 0.25), relative=True)

# We want the author to be in the center, 30px under the title
# We can set as pixels
top = background.h * 0.25 + title.h + 30
left = (background.w - author.w) / 2
author = author.with_position((left, top))

# We want the copyright to be 30px before bottom
copyright = copyright.with_position(("center", background.h - copyright.h - 30))

# Finally, we want the logo to be in the center, but to drop as time pass
# We can do so by setting position as a function that take time as argument,
# a lot like frame_function
top = (background.h - logo.h) / 2
logo = logo.with_position(lambda t: ("center", top + t * 30))

# We write the result
final_clip = CompositeVideoClip([background, title, author, copyright, logo])
final_clip.write_videofile(output_file)
