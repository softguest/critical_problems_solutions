import React from "react";

type EditorBlock = {
  id: string;
  type: string;
  data: any;
};

interface EditorRendererProps {
  blocks: EditorBlock[];
}

export const EditorRenderer: React.FC<EditorRendererProps> = ({ blocks }) => {
  return (
    <div className="editor-content space-y-4">
      {blocks.map((block) => {
        const { id, type, data } = block;

        switch (type) {
          case "header":
            const HeaderTag = `h${data.level}` as keyof JSX.IntrinsicElements;
            return <HeaderTag key={id}>{data.text}</HeaderTag>;

          case "paragraph":
            return <p key={id} dangerouslySetInnerHTML={{ __html: data.text }} />;

          case "code":
            return (
              <pre key={id} className="bg-gray-100 p-2 rounded overflow-x-auto">
                <code>{data.code}</code>
              </pre>
            );

          case "checklist":
            return (
              <ul key={id} className="space-y-1">
                {data.items.map((item: any, i: number) => (
                  <li key={i} className="flex items-center gap-2">
                    <input type="checkbox" checked={item.checked} readOnly />
                    <span dangerouslySetInnerHTML={{ __html: item.text }} />
                  </li>
                ))}
              </ul>
            );

          case "list":
            return data.style === "ordered" ? (
              <ol key={id} className="list-decimal list-inside">
                {data.items.map((item: string, i: number) => (
                  <li key={i} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              </ol>
            ) : (
              <ul key={id} className="list-disc list-inside">
                {data.items.map((item: string, i: number) => (
                  <li key={i} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              </ul>
            );

          case "image":
            return (
              <div key={id} className="text-center">
                <img src={data.file?.url} alt={data.caption || "Image"} className="mx-auto max-w-full rounded" />
                {data.caption && <p className="text-sm text-gray-600 mt-1">{data.caption}</p>}
              </div>
            );

          case "quote":
            return (
              <blockquote key={id} className="border-l-4 border-gray-400 pl-4 italic text-gray-700">
                <p dangerouslySetInnerHTML={{ __html: data.text }} />
                {data.caption && <footer className="text-sm text-gray-500 mt-1">— {data.caption}</footer>}
              </blockquote>
            );

          case "delimiter":
            return (
              <div key={id} className="flex justify-center my-4">
                <span className="text-2xl text-gray-400">***</span>
              </div>
            );

          case "embed":
            return (
              <div key={id} className="aspect-video">
                <iframe
                  className="w-full h-full rounded"
                  src={data.embed}
                  title={data.caption || "Embedded content"}
                  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            );

          default:
            return (
              <div key={id} className="text-red-500">
                Unsupported block type: <strong>{type}</strong>
              </div>
            );
        }
      })}
    </div>
  );
};
