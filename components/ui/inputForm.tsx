// ✅ Updated InputForm.tsx for modern trauma support chatbot
import { Loader2, Send } from "lucide-react";
import React, { ChangeEvent, FormEvent, useState } from "react";
import { ChatRequestOptions } from "ai";

type Props = {
  handleInputChange: (
    e: ChangeEvent<HTMLInputElement> | ChangeEvent<HTMLTextAreaElement>
  ) => void;
  handleSubmit: (
    e: FormEvent<HTMLFormElement>,
    chatRequestOptions?: ChatRequestOptions | undefined
  ) => void;
  input: string;
  isLoading: boolean;
  stop: () => void;
};

const InputForm = ({
  handleInputChange,
  handleSubmit,
  input,
  isLoading,
  stop,
}: Props) => {
  const [images, setImages] = useState<string[]>([]);

  const handleImageSelection = async (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    if (!files) return;
    const imagePromises = Array.from(files).map(
      (file) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result?.toString() || "");
          reader.onerror = (error) => reject(error);
          reader.readAsDataURL(file);
        })
    );

    try {
      const base64Strings = await Promise.all(imagePromises);
      setImages((prev) => [...prev, ...base64Strings]);
    } catch (error) {
      console.error("Error reading image:", error);
    }
  };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        handleSubmit(event, {
          data: {
            images: JSON.stringify(images),
          },
        });
      }}
      className="w-full flex gap-2 items-center px-2"
    >
      <input
        className="hidden"
        id="fileInput"
        type="file"
        accept="image/*"
        multiple
        onChange={handleImageSelection}
      />

      <input
        type="text"
        placeholder={isLoading ? "Responding..." : "Share your thoughts safely..."}
        value={input}
        disabled={isLoading}
        onChange={handleInputChange}
        className="flex-1 rounded-full py-3 px-5 bg-white border border-indigo-200 text-indigo-700 placeholder:text-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-300 disabled:opacity-60"
      />

      <button
        type="submit"
        className="rounded-full bg-indigo-500 hover:bg-indigo-600 transition text-white p-2 w-10 h-10 flex items-center justify-center"
      >
        {isLoading ? (
          <Loader2
            onClick={stop}
            className="animate-spin h-5 w-5"
          />
        ) : (
          <Send className="h-5 w-5" />
        )}
      </button>
    </form>
  );
};

export default InputForm;
