import ButtonGradient from './assets/svg/ButtonGradient'
import Benefits from './components/Benefits';
import Header from './components/Header';
import Hero from './components/Hero';
import Footer from './components/Footer';
import Usage from './components/Usage';
import Pricing from './components/Pricing';
import Overview from './components/Overview';
import Alphaedu from './components/Alphaedu';

const App = () => {
  return (
      <>
        <div className='pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden'>
          <Header />
          <Hero />
          <Overview />
          <Benefits />
          <Usage />
          <Alphaedu />
          <Pricing />
          <Footer />
        </div>
        <ButtonGradient />
      </>
  );
};

export default App
