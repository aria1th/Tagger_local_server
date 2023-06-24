# What is even this?

Using several models, we can think the following pipeline - if we want to change something to another one, "cat" to "dog", use segmentation, then generate mask, inpaint.

But sometimes we want to do batch... or run server and 'get' response to not load the model every time, or collaborate.

main api.py runs server for tagger, and accepts image, returns tags from it.

text_mask uses Grounding SAM for segmentation with prompt, returns mask (black-white if without json, else with labels).

Then with mask, you can send it to stable diffusion webui API, to get it inpainted with tags!
