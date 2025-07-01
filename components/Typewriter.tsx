import React, { useEffect, useState } from "react";

type Props = {
  text: string;
  speed?: number;
};

const Typewriter = ({ text, speed = 20 }: Props) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setDisplayedText((prev) => prev + text[i]);
      i++;
      if (i >= text.length) clearInterval(interval);
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return <div>{displayedText}</div>;
};

export default Typewriter;
