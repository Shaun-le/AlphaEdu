import Tagline from "./Tagline"
const Heading = ({ className, title, tag, text}) => {
  return (
    <div
        className={`${className} max-w-[50rem] mx-auto mb-12 lg:mb-20 md:text-center`}
    >
        {tag && <Tagline className="mb-4 md:justify-center">{tag}</Tagline>}
        {title && <h2 className="h2">{title}</h2>}
        {text && <p className="mt-4 text-n-3">{text}</p>}
    </div>
  )
}

export default Heading