---

pagetitle: "{{ replace .Name "-" " " | title }} | Chief Blog"
title: {{ replace .Name "-" " " | title }}
date: {{ .Date }}
featuredImage:
description: Description Here
Tags: 
- Tags
Categories:
- Category
draft: true
summary: Summary Here

---

## Title Here

Start Here

{{< figure src="" title="">}}
{{< gist url=>/1/2/の1と2をここにspaceで分ける >}}
{{< youtube "v="から始まるid >}}

```
def init():
    code = "Code goes here"
    return code
```