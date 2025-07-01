import EditorJSHTML from 'editorjs-html';
import { OutputData } from '@editorjs/editorjs';
import DOMPurify from 'dompurify';

const edjsParser = EditorJSHTML();

export function renderEditorContent(data?: OutputData): string {
  if (!data || !Array.isArray(data.blocks)) {
    return '<p>No content available</p>';
  }

  // Ensure .parse returns an array of { content: string }
  const parsed = edjsParser.parse(data); // should be an array

  if (!Array.isArray(parsed)) {
    console.warn("EditorJSHTML.parse did not return an array.");
    return '<p>Error rendering content</p>';
  }

  const htmlString = parsed.map((block: { content: string }) => block.content).join('');
  return DOMPurify.sanitize(htmlString);
}



