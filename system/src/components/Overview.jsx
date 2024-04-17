import Section from "./Section"
import Heading from "./Heading"
import { GradientLight } from "./design/Benefits";
import { collabContent, fileTypes } from "../constants";
import { check } from "../assets";
import { LeftCurve, RightCurve } from './design/Collaboration'
import CompanyLogos from "./CompanyLogos";

const Overview = () => {
  return (
    <Section crosses id='overview'>
      <Heading
          className="md:max-w-md lg:max-w-2xl"
          title="Introducing AlphaEdu"
          text="AlphaEdu is a product of AlphaLab, a research group specializing in Natural Language Processing (NLP). Developed with the aim of enhancing education in Vietnam, it serves as a valuable tool for both teachers and students, facilitating the creation of comprehensive review questions covering the material they've learned."
        />
      <div className="container lg:flex">
        <div className="max-w-[25rem]">
          <h2 className="h2 mb-4 md:mb-8">Diverse types of questions</h2>
          <ul className="max-w-[22rem] mb-10 md:mb-14">
            {collabContent.map((item) => (
              <li className="mb-3 py-3" key={item.id}>
                <div className="flex items-center">
                  <img 
                    src={check} width={24} height={24} alt='check'
                  />
                  <h6 className="body-2 ml-5">{item.title}</h6>
                </div>
                {item.question && (
                    <p className="body-2 mt-2 text-n-4">{item.question}</p>
                )}
                {item.mrc && (
                    <p className="body-2 text-n-4">{item.mrc}</p>
                )}
                {item.answer && (
                    <p className="body-2 text-amber-300">{item.answer}</p>
                )}
              </li>
            ))}
          </ul>
        </div>

        <div className="lg:ml-auto xl:w-[38rem] mt-4">
          <p className="body-2 mb-6 text-n-4 md:mb-16 lg:mb-32 lg:w-[22rem] lg:mx-auto">
          Share the topics or content you want to review, and I'll create a set of related questions for you
          </p>
        <div className="relative left-1/2 flex w-[22rem] aspect-square border
        border-n-6 rounded-full -translate-x-1/2 scale:75 md:scale-100">
            <div className="flex w-60 aspect-square m-auto border border-n-6
            rounded-full">
              <div className="w-[6rem] aspect-square m-auto p-[0.2rem] bg-conic-gradient
              rounded-full">
                <div className="flex items-center justify-center w-full h-full
                bg-n-8 rounded-full">
                  <img 
                    src='./src/assets/logo/ASymbol.png'
                    width={48}
                    height={48}
                  />
                </div>
              </div>
            </div>
            <ul>
              {fileTypes.map((app, index) => (
                <li key={app.id} className={`absolute top-0 left-1/2 h-1/2 -ml-[1.6rem] origin-bottom
                rotate-${index*45}`}>
                  <div className={`relative -top-[1.6rem] flex w-[3.2rem] h-[3.2rem] bg-n-7
                  border border-n-1/15 rounded-xl -rotate-${index*45}`}>
                    <img
                      className="m-auto"
                      width={app.width}
                      height={app.height}
                      src={app.icon}
                      alt={app.title}
                    />
                  </div>
                </li>
              ))}
            </ul>
            <LeftCurve />
            <RightCurve />
          </div>
        </div>
      </div>
      <div className="container pt-15">
        <CompanyLogos />
      </div>
    </Section>
  )
}

export default Overview