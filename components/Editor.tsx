'use client';

import { useEffect, useRef } from 'react';
import EditorJS from '@editorjs/editorjs';
import Header from '@editorjs/header';
import List from '@editorjs/list';
import Paragraph from '@editorjs/paragraph';

interface EditorProps {
  onChange: (data: any) => void;
}

export default function Editor({ onChange }: EditorProps) {
  const editorRef = useRef<EditorJS | null>(null);
  const holderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!editorRef.current) {
      editorRef.current = new EditorJS({
        holder: holderRef.current!,
        tools: {
          header: Header,
          list: List,
          paragraph: Paragraph,
        },
        onChange: async () => {
          const output = await editorRef.current!.save();
          onChange(output);
        },
      });
    }

    return () => {
      if (editorRef.current && typeof editorRef.current.destroy === 'function') {
        editorRef.current.destroy();
        editorRef.current = null;
      }
    };
  }, [onChange]);

  return <div ref={holderRef} className="border rounded-lg p-4" />;
}