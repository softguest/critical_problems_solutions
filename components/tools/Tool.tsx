import Header from "@editorjs/header"
import List from "@editorjs/list"
import Checklist from "@editorjs/checklist"
import Paragraph from '@editorjs/paragraph';
import ImageTool from '@editorjs/image';
import CodeTool from '@editorjs/code';
import Embed from '@editorjs/embed';

export const EDITOR_JS_TOOLS = {
    header: Header,
    paragraph: Paragraph,
    list: List,
    checklist: Checklist,
    code: CodeTool,
    embed: Embed,
    image: {
      class: ImageTool,
      config: {
        uploader: {
          async uploadByFile(file: File) {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/upload", {
              method: "POST",
              body: formData,
            });

            const { url } = await res.json();
            return {
              success: 1,
              file: { url },
            };
          },
        },
      },
    },
}