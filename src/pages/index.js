import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Overview of the Humanoid Robotics & Physical AI Course Book">
      <main>
        <section
          className="hero hero--primary"
          style={{
            backgroundImage: 'url(/img/hero-background.jpg)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            padding: '80px 20px',
            textAlign: 'center',
            minHeight: '500px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <div
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              padding: '40px',
              borderRadius: '8px',
              maxWidth: '800px'
            }}
          >
            <h1
              className="hero__title"
              style={{
                fontSize: '2.5rem',
                marginBottom: '20px',
                color: 'white'
              }}
            >
              {siteConfig.title}
            </h1>
            <p
              className="hero__subtitle"
              style={{
                fontSize: '1.2rem',
                marginBottom: '30px',
                color: '#ddd'
              }}
            >
              {siteConfig.tagline}
            </p>
            <div>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                Read the Book - Start Learning
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}