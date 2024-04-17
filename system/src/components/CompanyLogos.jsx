import { companyLogos } from "../constants"

const CompanyLogos = ({ className }) => {
  return (
    <div className={className}>
        <h5 className="tagline mb-6 text-center text-n-1/50">
        Currently, we support the following subjects: 
        </h5>
        <ul className="flex">
            {companyLogos.map((logo, index) => (
                <li className="flex items-center justify-center flex-1 h-[8.5rem]" key={index}>
                    <img 
                        className="border border-n-1/15 rounded-2xl "
                        src={logo} width={64} height={64} alt={logo}
                    />
                </li>
            ))}
        </ul>
    </div>
  )
}

export default CompanyLogos